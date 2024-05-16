import os.path as osp
import os
import cv2
import numpy as np
from glob import glob
import torch

from sg.matching import Matching
from sg.utils import frame2tensor
from sg.utils import make_matching_plot_color, draw_SPMatches
from loftr.loftr import LoFTR, default_cfg
from loftr.utils.plotting import make_matching_plot_color_Loftr
from sklearn.metrics.cluster import mutual_info_score
import matplotlib.cm as cm

import argparse

torch.set_grad_enabled(False)
cv2.ocl.setUseOpenCL(False)

W = 640
H = 480


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': 8,
        'keypoint_threshold': 0.001,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 100,
        'match_threshold': 0.1,
    }
}
matching = Matching(config).eval().to(device)
matcher = LoFTR(config=default_cfg)
image_type = 'outdoor'
if image_type == 'indoor':
    matcher.load_state_dict(torch.load("loftr/weights/indoor_ds_new.ckpt")['state_dict'])
elif image_type == 'outdoor':
    matcher.load_state_dict(torch.load("loftr/weights/outdoor_ds.ckpt")['state_dict'])
else:
    raise ValueError("Wrong image_type is given.")
matcher = matcher.eval().to(device)

def nn_match_two_way(desc1, desc2, nn_thresh=0.7):
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    dmat = np.dot(desc1, desc2.T)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    keep = scores < nn_thresh
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    m_idx1 = np.arange(desc1.shape[0])[keep]
    m_idx2 = idx
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


def sp_stitch_two(im11, im22):
    psd_img_1 = im11
    psd_img_2 = im22
    im11 = cv2.cvtColor(im11, cv2.COLOR_BGR2GRAY)
    im22 = cv2.cvtColor(im22, cv2.COLOR_BGR2GRAY)
    im11 = cv2.resize(im11.astype('float32'), (W, H))
    im22 = cv2.resize(im22.astype('float32'), (W, H))
    inp0 = frame2tensor(im11, device)
    inp1 = frame2tensor(im22, device)
    pred, pred0, pred1 = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    des_sp0, des_sp1 = (pred0['descriptors'][0].t()).cpu().numpy(), (pred1['descriptors'][0].t()).cpu().numpy()
    sp_matches = nn_match_two_way(des_sp0, des_sp1, nn_thresh=0.8)
    better = sp_matches[0:2, :]
    confidence_sp = 1 - pow(sp_matches[2, :], 2)
    color_sp = cm.jet(confidence_sp)
    img_sp = draw_SPMatches(psd_img_1, psd_img_2, kpts0, kpts1, better, color_sp)
    _, num_match = better.shape
    mkpts0 = np.float32([kpts0[int(i)] for i in better[0, :]])
    mkpts1 = np.float32([kpts1[int(i)] for i in better[1, :]])

    if num_match > 10:
        (HH, status) = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, ransacReprojThreshold=5.0)
        if HH[0][0] <= 3 and HH[0][0] >= 0.4 and HH[1][1] <= 3 and HH[1][1] >= 0.4 and abs(HH[0][-1]) < W and abs(
                HH[1][-1]) < H:
            return True, HH, num_match, img_sp
        return False, None, num_match, img_sp
    else: return False, None, num_match, img_sp


def sg_stitch_two(im11, im22):
    psd_img_1 = im11
    psd_img_2 = im22
    im11 = cv2.cvtColor(im11, cv2.COLOR_BGR2GRAY)
    im22 = cv2.cvtColor(im22, cv2.COLOR_BGR2GRAY)
    im11 = cv2.resize(im11.astype('float32'), (W, H))
    im22 = cv2.resize(im22.astype('float32'), (W, H))

    inp0 = frame2tensor(im11, device)
    inp1 = frame2tensor(im22, device)
    pred, pred0, pred1 = matching({'image0': inp0, 'image1': inp1})

    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    mdesc0, mdesc1 = pred['descriptors0'], pred['descriptors1']
    matches = pred['matches0']
    confidence = pred['matching_scores0']
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    m_color = cm.jet(confidence[valid])
    out = make_matching_plot_color(
        psd_img_1, psd_img_2, kpts0, kpts1, mkpts0, mkpts1, m_color, [],
        path=None, show_keypoints=False, small_text=[])
    if len(mkpts0) > 10:
        (HH, status) = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, ransacReprojThreshold=5.0)
        return True, HH, len(mkpts0), out
    else: return False, None, 0, out


def loftr_stitch_two(im11, im22):
    psd_img_1 = im11
    psd_img_2 = im22
    im11 = cv2.cvtColor(im11, cv2.COLOR_BGR2GRAY)
    im22 = cv2.cvtColor(im22, cv2.COLOR_BGR2GRAY)
    im11 = cv2.resize(im11.astype('float32'), (W, H))
    im22 = cv2.resize(im22.astype('float32'), (W, H))

    inp0 = frame2tensor(im11, device)
    inp1 = frame2tensor(im22, device)

    batch = {'image0': inp0, 'image1': inp1}
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    color = cm.jet(mconf, alpha=0.7)
    out = make_matching_plot_color_Loftr(mconf, 0.3,
                                         psd_img_1, psd_img_2, mkpts0, mkpts1, color, [],
                                         path=None, show_keypoints=False, small_text=[], WW=W, HH=H)
    if len(mkpts0) > 10:
        (HH, status) = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, ransacReprojThreshold=5.0)
        # print(HH)
        if HH[0][0] <= 3 and HH[0][0] >= 0.35 and HH[1][1] <= 3 and HH[1][1] >= 0.35 and abs(HH[0][-1]) < W and abs(HH[1][-1]) < H:
            return True, HH, len(mkpts0), out
        else:
            return False, None, len(mkpts0), out
    else: return False, None, 0, out


def sift_stitch_two(im11, im22):

    im11 = cv2.resize(im11.astype('uint8'), (W, H))
    im22 = cv2.resize(im22.astype('uint8'), (W, H))
    psd_img_1 = im11
    psd_img_2 = im22
    im11 = cv2.cvtColor(im11, cv2.COLOR_BGR2GRAY)
    im22 = cv2.cvtColor(im22, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kpp1, des1 = sift.detectAndCompute(im11, None)  # des是描述子
    kpp2, des2 = sift.detectAndCompute(im22, None)  # des是描述子
    kp1 = np.float32([kp.pt for kp in kpp1])
    kp2 = np.float32([kp.pt for kp in kpp2])
    # print(des1, des2)
    if des1 is None or des2 is None:
        return False, None, 0
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = []

    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            good.append((m[0].trainIdx, m[0].queryIdx))

    good_point = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_point.append([m])
    num_match = len(good)
    ptsA = np.float32([kp1[i] for (_, i) in good])
    ptsB = np.float32([kp2[i] for (i, _) in good])
    out = cv2.drawMatchesKnn(psd_img_1, kpp1, psd_img_2, kpp2, good_point, None, flags=2)
    if num_match > 10:
        (HH, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, ransacReprojThreshold=5.0)
        if HH[0][0] <= 2 and HH[0][0] >= 0.5 and HH[1][1] <= 2 and HH[1][1] >= 0.5 and abs(HH[0][-1]) < W and abs(HH[1][-1]) < H:
            return True, HH, num_match, out
        else:
            return False, None, num_match, out
    else:
        return False, None, num_match, out


def match_ratio(name1, name2):
    cnt = 0
    for i in range(len(name1)):
        if name1[i] != name2[i]:
            cnt += 1
    return cnt


def hh_stitch(image_list, HH_list, match_method):
    # print(len(image_list), len(HH_list))
    cur_index = len(HH_list)
    for i in range(cur_index + 1, len(image_list)):

        if match_method == 'sift':
            status, HH, num_match, img_match = sift_stitch_two(image_list[i - 1], image_list[i])
        if match_method == 'sp':
            status, HH, num_match, img_match = sp_stitch_two(image_list[i - 1], image_list[i])
        if match_method == 'sg':
            status, HH, num_match, img_match = sg_stitch_two(image_list[i - 1], image_list[i])
        if match_method == 'loftr':
            status, HH, num_match, img_match = loftr_stitch_two(image_list[i - 1], image_list[i])

        if not status:
            return 1, [], [], num_match, img_match
        HH_list.append(HH)

    return 0, image_list, HH_list, num_match, img_match
    # return 0, image_curr

def treat_list(image_list, homo_list):

    center = len(image_list) // 2

    HH_bias = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(0, center):
        HH_bias = np.dot(HH_bias, homo_list[i])
    HH_bias = np.linalg.inv(HH_bias)
    HH_cur_to_first = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # image_center = image_list[center]

    HH_use_list = []
    min_w = 1000
    min_h = 1000
    max_w = -1000
    max_h = -1000
    for i in range(len(image_list)):
        HH_cur = np.dot(HH_bias, HH_cur_to_first)
        f1 = np.dot(HH_cur, np.array([0, 0, 1]))
        f1 = f1 / f1[-1]
        f2 = np.dot(HH_cur, np.array([W, 0, 1]))
        f2 = f2 / f2[-1]
        f3 = np.dot(HH_cur, np.array([0, H, 1]))
        f3 = f3 / f3[-1]
        f4 = np.dot(HH_cur, np.array([W, H, 1]))
        f4 = f4 / f4[-1]
        # print(i, HH_cur)
        HH_use = HH_cur
        HH_use_list.append(HH_use)
        min_w = min(min_w, min([f1[0], f2[0], f3[0], f4[0]]))
        min_h = min(min_h, min([f1[1], f2[1], f3[1], f4[1]]))
        max_w = max(max_w, max([f1[0], f2[0], f3[0], f4[0]]))
        max_h = max(max_h, max([f1[1], f2[1], f3[1], f4[1]]))
        if i < len(image_list) - 1:
            HH_cur_to_first = np.dot(HH_cur_to_first, homo_list[i])
    max_w_int = int(max_w + 1)
    max_h_int = int(max_h + 1)
    min_w_int = int(min_w - 1)
    min_h_int = int(min_h - 1)
    # print(max_w_int, max_h_int, min_w_int, min_h_int)
    newW = - min_w_int + max_w_int
    newH = - min_h_int + max_h_int
    # print(newW, newH)
    newH = min(newH, 2 * H)
    newW = min(newW, 2 * W)
    try:
        pano = np.zeros([newH, newW, 3], np.uint8)
        image_cur_list = []
        for i in range(len(image_list)):
            HH_use = HH_use_list[i]
            HH_use[0][-1] += - min_w_int
            HH_use[1][-1] += - min_h_int
            image_cur = cv2.warpPerspective(image_list[i], HH_use, (newW, newH))
            image_cur_list.append(image_cur)
            pano = pano + image_cur / 2
            if i < len(image_list) - 1:
                HH_cur_to_first = np.dot(HH_cur_to_first, homo_list[i])
        aa = np.sum(image_cur_list[0] * image_cur_list[1], axis=2) > 0
        # aa = np.sum(image_cur_list[0], axis=2) > 0
        # print(aa.shape, image_cur_list[0].shape, image_cur_list[1].shape)
        img1 = np.mean(image_cur_list[0][aa], axis=-1)
        img2 = np.mean(image_cur_list[1][aa], axis=-1)
        img1 = np.reshape(img1, -1)
        img2 = np.reshape(img2, -1)
        # print(img1.shape, img2.shape)
        if np.sum(aa)>0:
            MI = mutual_info_score(img1, img2)
        else:
            MI = 0

    except:
        pano = np.zeros([W, H, 3], np.float32)
        return pano, 0

    return pano, MI


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='sg', help='sift, sp, sg, loftr')
    parser.add_argument('--root', default='/home/fsh/stitching_and_registration/input_data/stitch_data/stitch')
    parser.add_argument('--outfile', default='/home/fsh/stitching_and_registration/output_data/stitch_out')
    args = parser.parse_args()

    match_method = args.method
    root = args.root
    out_root = args.outfile

    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    image_list = []
    notfirst = 0
    cnt_pinjie = 0
    MI_list = []
    num_match_list = []
    name_list = []
    max_row_sum = 500
    total = 0
    success = 0

    out_path = osp.join(out_root, 'cross')

    image_dir1 = osp.join(root, '智能巡视点位数据/三站主变点位/东港站主变点位/')
    image_dir2 = osp.join(root, '智能巡视点位数据/三站主变点位/傅疃站主变点位/')
    image_dir3 = osp.join(root, '智能巡视点位数据/三站主变点位/岚山站主变点位/')

    image_dir_pair = []
    if not osp.exists(out_path):
        os.makedirs(out_path)
    fpaths1 = sorted(glob(osp.join(image_dir1, '*')))
    fpaths2 = sorted(glob(osp.join(image_dir2, '*')))
    fpaths3 = sorted(glob(osp.join(image_dir3, '*')))
    fpaths = fpaths1 + fpaths2 + fpaths3
    # print(fpaths)
    file_cnt = 0
    for file in fpaths:
        # print(file)
        image_hw_path = sorted(glob(osp.join(file, '红外', '*.jpg')))
        image_kjg_path = sorted(glob(osp.join(file, '可见光', '*.jpg')))
        if len(image_hw_path) == 1 and len(image_kjg_path) == 1:
            file_cnt += 1
            img1 = cv_imread(image_hw_path[0])
            img1 = cv2.resize(img1.astype('float32'), (W, H))[:, :, :3]
            img2 = cv_imread(image_kjg_path[0])
            img2 = cv2.resize(img2.astype('float32'), (W, H))
            image_dir_pair.append([img1, img2])

    cnt = 0
    draw_list = [0, 55]
    for i in draw_list:
        print(i)
        image_list = image_dir_pair[i]
        homo_list = []
        (status, image_list, homo_list, num_match, img_match) = hh_stitch(image_list, homo_list, match_method)
        image_name = str(i) + '_' + match_method + '_' + str(len(image_list)) + '.jpg'

        if status == cv2.Stitcher_OK:
            cnt += 1
            pano, MI = treat_list(image_list, homo_list)
            cnt_pinjie += 1
            print("拼接成功.")
            image_path = osp.join(out_path, image_name)
            cv2.imwrite(image_path, pano)
            ori_dir_name = str(i)
            ori_dir = osp.join(out_path, ori_dir_name)
            if not osp.exists(ori_dir):
                os.makedirs(ori_dir)
            for j in range(len(image_list)):
                image_path = osp.join(ori_dir, str(i + j) + '_.jpg')
                cv2.imwrite(image_path, image_list[j])
        else:
            print("拼接失败.")





