import cv2
import time
import numpy as np
import matplotlib.cm as cm
import torch
import os
import random
from models.matching import Matching
from scipy import optimize
import os.path as osp
from glob import glob
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
from RIFT_no_rotation_invariance import RIFT
from phasepack import phasecong
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane, make_matching_plot_color,
                          scale_intrinsics)

import sklearn.metrics as skm

from sklearn.metrics.cluster import mutual_info_score



def make_matching_plot_color_Loftr(conf, show_THR, image0, image1, kpts0, kpts1, color, text, path=None,
                             show_keypoints=True, margin=0,
                             opencv_display=False, opencv_title='',
                             small_text=[], WW=640, HH=480):
    # H0, W0, _ = image0.shape
    # H1, W1, _ = image1.shape

    [H0, W0, H1, W1] = [HH, WW, HH, WW]

    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0 + margin:, :] = image1
    # out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]

    img_hw = image0
    img_kjg = image1
    for p0, p1, c, cc in zip(mkpts0, mkpts1, color, conf):
        # print(p0, p1)
        if cc < show_THR:
            continue
        x0 = p0[0]
        y0 = p0[1]
        x1 = p1[0]
        y1 = p1[1]
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # print(c)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    # if path is not None:
    #     cv2.imwrite(str(path), out)

    # if opencv_display:
    #     cv2.imshow(opencv_title, out)
    #     cv2.waitKey(1)

    return out



def hxx_forward(x, y):
    return skm.mutual_info_score(x, y)

def Point_Coordinates(keypoints1, keypoints2):
    x1 = []
    x2 = []
    l1 = len(keypoints1)
    l2 = len(keypoints2)

    for i in range(l1):
        tuple_x1 = keypoints1[i].pt
        x1.append(tuple_x1)
    for i in range(l2):
        tuple_x2 = keypoints2[i].pt
        x2.append(tuple_x2)

    return np.array(x1, dtype=float), np.array(x2, dtype=float)

def hxx(x, y):
    size = x.shape[-1]
    px = np.histogram(x, 256, (0, 255))[0] / size
    py = np.histogram(y, 256, (0, 255))[0] / size
    hx = - np.sum(px * np.log(px + 1e-8))
    hy = - np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
    hxy /= (1.0 * size)
    hxy = - np.sum(hxy * np.log(hxy + 1e-8))

    r = hx + hy - hxy
    return r


def merge_sg(mode_path, out_path, im1, im2, device, W, H, psd_img_1, psd_img_2, working_dir, im_file, matching):
    image0, inp0, scales0 = read_image(
        im1, device, [W, H], 0, None)
    image1, inp1, scales1 = read_image(
        im2, device, [W, H], 0, None)
    # baseline
    pred, pred0, pred1 = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches = pred['matches0']
    confidence = pred['matching_scores0']
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])
    # print(confidence_sp, confidence)
    # print(color_sp, color)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    # text = []
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    stem0 = 0
    stem1 = 0

    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {:06}:{:06}'.format(stem0, stem1),
    ]

    out = make_matching_plot_color(
        psd_img_1, psd_img_2, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=None, small_text=small_text)
    out_file = working_dir + "/" + out_path + "/" + mode_path + "/match/"
    if not os.path.exists(out_file):
        os.makedirs(out_file)

    cv2.imwrite(out_file + im_file, out)

    if mode_path == "mimout":
        psd_img_11 = cv2.imread(im1, cv2.IMREAD_COLOR)
        psd_img_22 = cv2.imread(im2, cv2.IMREAD_COLOR)
        psd_img_11 = cv2.resize(psd_img_11, (W, H))
        psd_img_22 = cv2.resize(psd_img_22, (W, H))
        out = make_matching_plot_color(
            psd_img_11, psd_img_22, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=None, small_text=small_text)
        out_file = working_dir + "/" + out_path + "/" + mode_path + "/mimmatch/"
        if not os.path.exists(out_file):
            os.makedirs(out_file)
        cv2.imwrite(out_file + im_file, out)
    rate = 2*len(mkpts0) / (len(kpts0)+len(kpts1))
    # print(rate)
    if len(mkpts0) <= 10:
        fused = psd_img_2 // 2 + psd_img_1 // 2
        x = np.reshape(psd_img_2, -1)
        y = np.reshape(psd_img_1, -1)
        MI = hxx_forward(x, y)
        NCC = np.mean(np.multiply((psd_img_1 - np.mean(psd_img_1)), (psd_img_2 - np.mean(psd_img_2)))) / (np.std(psd_img_1) * np.std(psd_img_2))
        # print(MI)
        MI = mutual_info_score(x, y)
        return len(mkpts0), MI, rate
    (HH, status) = cv2.findHomography(mkpts0, mkpts1, cv2.RHO, ransacReprojThreshold=5.0)
    result = cv2.warpPerspective(psd_img_1, HH, (max(psd_img_1.shape[1], psd_img_2.shape[1]), psd_img_2.shape[0]))
    fused = psd_img_2 // 2 + result // 2

    img_sg = fused
    out_file = working_dir + "/" + out_path + "/" + mode_path + "/merge/"
    # print('\nWriting image to {}'.format(out_file))
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    cv2.imwrite(out_file + im_file, img_sg)
    if mode_path == "mimout":
        psd_img_11 = cv2.imread(im1, cv2.IMREAD_COLOR)
        psd_img_22 = cv2.imread(im2, cv2.IMREAD_COLOR)
        psd_img_11 = cv2.resize(psd_img_11, (W, H))
        psd_img_22 = cv2.resize(psd_img_22, (W, H))

        resultmim = cv2.warpPerspective(psd_img_11, HH, (max(psd_img_11.shape[1], psd_img_22.shape[1]), psd_img_22.shape[0]))
        fusedmim = psd_img_22 // 2 + resultmim // 2
        img_sg_mim = fusedmim
        out_file = working_dir + "/" + out_path + "/" + mode_path + "/mimmerge/"
        # print('\nWriting image to {}'.format(out_file))
        if not os.path.exists(out_file):
            os.makedirs(out_file)

        cv2.imwrite(out_file + im_file, img_sg_mim)
    # x = np.reshape(psd_img_2, -1)
    # y = np.reshape(result, -1)
    # MI = hxx_forward(x, y)
    # print(result.shape)
    aa = np.sum(result, axis=2)>0
    # print(aa.shape,aa)
    img1 = np.mean(result[aa], axis=1)
    img2 = np.mean(psd_img_2[aa], axis=1)
    # print(img1.shape, img2.shape)
    NCC = np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (
                np.std(img1) * np.std(img2))
    MI = mutual_info_score(img1, img2)
    # print(MI)
    # print(NCC)
    return len(mkpts0), MI, rate


def kp2point(kps):
    # print(len(kps))
    kp2 = []
    for kp in kps:
        # print(kp.pt)
        a, b = kp.pt
        kp2.append((b,a))

    return np.array(kp2)


def merge_sift(mode_path, out_path, im1, im2, device, W, H, psd_img_1, psd_img_2, working_dir, im_file, matching):
    if mode_path == 'orgmimout':
        m1, __, __, __, PC1, eo1, __ = phasecong(img=psd_img_1, nscale=4, norient=6, minWaveLength=3, mult=1.6,
                                                sigmaOnf=0.75, g=3, k=1)
        m2, __, __, __, PC2, eo2, __ = phasecong(img=psd_img_2, nscale=4, norient=6, minWaveLength=3, mult=1.6,
                                                sigmaOnf=0.75, g=3, k=1)

        # 将提取到的特征转为unit8格式
        psd_img_11 = cv2.imread(im1, cv2.IMREAD_COLOR)
        psd_img_22 = cv2.imread(im2, cv2.IMREAD_COLOR)
        psd_img_11 = cv2.resize(psd_img_11, (W, H))
        psd_img_22 = cv2.resize(psd_img_22, (W, H))
        m1, m2 = map(lambda img: (img.astype(np.float) - img.min()) / (img.max() - img.min()), (m1, m2))
        cm1 = m1 * 255
        cm2 = m2 * 255

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(np.uint8(cm1), None)  # des是描述子
        kp2, des2 = sift.detectAndCompute(np.uint8(cm2), None)  # des是描述子
        bf = cv2.BFMatcher()
        if des1 is None or des2 is None:
            return 0
        matches = bf.knnMatch(des1, des2, k=2)
        # 调整ratio
        good = []
        dist = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good.append([m])
                dist.append(m.distance)
        num_match = len(good)
        img_match = cv2.drawMatchesKnn(psd_img_1, kp1, psd_img_2, kp2, good, None, flags=2)
        out_file = working_dir + "/" + out_path + "/" + mode_path + "/match/"
        if not os.path.exists(out_file):
            os.makedirs(out_file)
        cv2.imwrite(out_file + im_file, img_match)

    else:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(psd_img_1, None)  # des是描述子
        kp2, des2 = sift.detectAndCompute(psd_img_2, None)  # des是描述子
        bf = cv2.BFMatcher()
        # print(des1, des2)
        if des1 is None or des2 is None:
            return 0
        matches = bf.knnMatch(des1, des2, k=2)
        # 调整ratio
        good = []
        dist = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good.append([m])
                dist.append(m.distance)
        num_match = len(good)
        img_match = cv2.drawMatchesKnn(psd_img_1, kp1, psd_img_2, kp2, good, None, flags=2)
    out_file = working_dir + "/" + out_path + "/" + mode_path + "/match/"
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    cv2.imwrite(out_file + im_file, img_match)
    return num_match


def nn_match_two_way(desc1, desc2, nn_thresh=0.7):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    # assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1, desc2.T)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[0])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


def draw_SPMatches(imageA, imageB, kpsA, kpsB, matches, color):
    # (hA, wA) = imageA.shape[:2]
    # (hB, wB) = imageB.shape[:2]
    # [hA, hB, wA, wB] = [480, 480, 640, 640]
    [hA, hB, wA, wB] = [360, 360, 480, 480]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB:, wA:] = imageB

    show_points = True
    if show_points:
        LA, _ = kpsA.shape
        LB, _ = kpsB.shape
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    _, L = matches.shape

    for i in range(L):
        c = color[i]
        trainIdx = int(matches[0][i])
        queryIdx = int(matches[1][i])
        c = c.tolist()
        ptA = (int(kpsA[trainIdx][0]), int(kpsA[trainIdx][1]))
        ptB = (int(kpsB[queryIdx][0]) + wA, int(kpsB[queryIdx][1]))
        cv2.line(vis, ptA, ptB, c, 1, lineType=cv2.LINE_AA)

    return vis


def merge_loftr(mode_path, out_path, im1, im2, device, W, H, psd_img_1, psd_img_2, working_dir, im_file, matching):
    image0, inp0, scales0 = read_image(
        im1, device, [W, H], 0, None)
    image1, inp1, scales1 = read_image(
        im2, device, [W, H], 0, None)
    # baseline
    batch = {'image0': inp0, 'image1': inp1}
    with torch.no_grad():
        matching(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    # Draw
    color = cm.jet(mconf, alpha=0.7)
    text = []
    out = make_matching_plot_color_Loftr(mconf, 0.3,
        psd_img_1, psd_img_2, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=True, small_text=[], WW=W, HH=H)

    out_file = working_dir + "/" + out_path + "/" + mode_path + "/match/"

    if not os.path.exists(out_file):
        os.makedirs(out_file)
    cv2.imwrite(out_file + im_file, out)

    if mode_path == "mimout":
        psd_img_11 = cv2.imread(im1, cv2.IMREAD_COLOR)
        psd_img_22 = cv2.imread(im2, cv2.IMREAD_COLOR)
        psd_img_11 = cv2.resize(psd_img_11, (W, H))
        psd_img_22 = cv2.resize(psd_img_22, (W, H))
        out = make_matching_plot_color_Loftr(mconf, 0.3,
            psd_img_11, psd_img_22, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=True, small_text=[], WW=W, HH=H)
        out_file = working_dir + "/" + out_path + "/" + mode_path + "/mimmatch/"
        if not os.path.exists(out_file):
            os.makedirs(out_file)
        cv2.imwrite(out_file + im_file, out)
    # rate = 2*len(mkpts0) / (len(kpts0)+len(kpts1))
    # print(rate)
    if len(mkpts0) <= 10:
        fused = psd_img_2 // 2 + psd_img_1 // 2
        x = np.reshape(psd_img_2, -1)
        y = np.reshape(psd_img_1, -1)
        # MI = hxx_forward(x, y)
        # NCC = np.mean(np.multiply((psd_img_1 - np.mean(psd_img_1)), (psd_img_2 - np.mean(psd_img_2)))) / (np.std(psd_img_1) * np.std(psd_img_2))
        MI = mutual_info_score(x, y)
        return len(mkpts0), MI
    # print(len(mkpts0), len(mkpts1))
    (HH, status) = cv2.findHomography(mkpts0, mkpts1, cv2.RHO, ransacReprojThreshold=5.0)
    # print(HH)
    if HH is None:
        fused = psd_img_2 // 2 + psd_img_1 // 2
        x = np.reshape(psd_img_2, -1)
        y = np.reshape(psd_img_1, -1)
        MI = hxx_forward(x, y)
        NCC = np.mean(np.multiply((psd_img_1 - np.mean(psd_img_1)), (psd_img_2 - np.mean(psd_img_2)))) / (
                    np.std(psd_img_1) * np.std(psd_img_2))
        # print(MI)
        MI = mutual_info_score(x, y)
        return len(mkpts0), MI
    result = cv2.warpPerspective(psd_img_1, HH, (max(psd_img_1.shape[1], psd_img_2.shape[1]), psd_img_2.shape[0]))
    fused = psd_img_2 // 2 + result // 2
    out_file = working_dir + "/" + out_path + "/" + mode_path + "/merge/"
    # print('\nWriting image to {}'.format(out_file))
    if not os.path.exists(out_file):
        os.makedirs(out_file)

    cv2.imwrite(out_file + im_file, fused)
    if mode_path == "mimout":
        psd_img_11 = cv2.imread(im1, cv2.IMREAD_COLOR)
        psd_img_22 = cv2.imread(im2, cv2.IMREAD_COLOR)
        psd_img_11 = cv2.resize(psd_img_11, (W, H))
        psd_img_22 = cv2.resize(psd_img_22, (W, H))

        resultmim = cv2.warpPerspective(psd_img_11, HH, (max(psd_img_11.shape[1], psd_img_22.shape[1]), psd_img_22.shape[0]))
        fusedmim = psd_img_22 // 2 + resultmim // 2
        out_file = working_dir + "/" + out_path + "/" + mode_path + "/mimmerge/"
        # print('\nWriting image to {}'.format(out_file))
        if not os.path.exists(out_file):
            os.makedirs(out_file)
        cv2.imwrite(out_file + im_file, fusedmim)

    aa = np.sum(result, axis=2)>0
    # print(aa.shape,aa)
    img1 = np.mean(result[aa], axis=1)
    img2 = np.mean(psd_img_2[aa], axis=1)
    # print(img1.shape, img2.shape)
    # NCC = np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (
    #             np.std(img1) * np.std(img2))
    MI = mutual_info_score(img1, img2)

    return len(mkpts0), MI
