import cv2
import time
import numpy as np
import matplotlib.cm as cm
import torch
import random
from models.matching import Matching
from scipy import optimize
import os.path as osp
from glob import glob
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane, make_matching_plot_color,
                          scale_intrinsics)
from RIFT_no_rotation_invariance import RIFT
from phasepack import phasecong
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from basemerge import make_matching_plot_color_Loftr
import torch
from PIL import Image
from GANmodels import Generator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.set_grad_enabled(False)

import sklearn.metrics as skm

import argparse
import pprint

from src.loftr import LoFTR, default_cfg
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "3"


def hxx_forward(x, y):
    return skm.mutual_info_score(x, y)


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


def kp2point(kps):
    # print(len(kps))
    kp2 = []
    for kp in kps:
        kp2.append(kp.pt)

    return np.array(kp2)


def get_path(path_current):
    path_count = 1
    path_current_split = path_current.split('/')
    path_want = path_current_split[0]
    path_last = path_current_split[-1]
    # print(path_last)
    for i in range(len(path_current_split) - 1 - path_count):
        j = i + 1
        path_want = path_want + '/' + path_current_split[j]
    return path_want, path_last


if __name__ == '__main__':

    print(cv2.__version__)
    # config = get_cfg_defaults()
    matcher = LoFTR(config=default_cfg)
    image_type = 'outdoor'
    if image_type == 'indoor':
        matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")['state_dict'])
    elif image_type == 'outdoor':
        matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
    else:
        raise ValueError("Wrong image_type is given.")
    matcher = matcher.eval().cuda()
    working_dir = osp.dirname(osp.abspath(__file__))
    working_dir, _ = get_path(working_dir)
    # data_path = 'data'
    j = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    ##### adjust your filename #####
    file = '/home/fsh/stitching_and_registration/input_data/demo_reg/demo/examples/1'
    out_dir = '/home/fsh/stitching_and_registration/output_data/demo_reg/demo/examples/1'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    show_THR = 0.3
    H = 360
    W = 480
    # if j < 200:
    #     continue
    # file_name = ''
    file_name = '20200708104657345'
    # file_name = '20200715141612995'
    # file_name = '20200715162544776'
    # file_name = '20200708104657345'

    #### enter input image names #####
    im1 = file + '/' + file_name + '_hw.jpg'
    im2 = file + '/' + file_name + '_kjg.jpg'


    psd_img_1 = cv2.imread(im1, cv2.IMREAD_COLOR)
    psd_img_2 = cv2.imread(im2, cv2.IMREAD_COLOR)
    psd_img_1 = cv2.resize(psd_img_1, (W, H))
    psd_img_2 = cv2.resize(psd_img_2, (W, H))
    # out_file = file + '/' + file_name + 'hw.jpg'
    # cv2.imwrite(out_file, psd_img_1)
    # out_file = file + '/' + file_name + 'kjg.jpg'
    # cv2.imwrite(out_file, psd_img_2)
    m1, __, __, __, __, eo1, __ = phasecong(img=psd_img_1, nscale=4, norient=6, minWaveLength=3, mult=1.6,
                                            sigmaOnf=0.75, g=3, k=1)
    m2, __, __, __, __, eo2, __ = phasecong(img=psd_img_2, nscale=4, norient=6, minWaveLength=3, mult=1.6,
                                            sigmaOnf=0.75, g=3, k=1)

    # 将提取到的特征转为unit8格式
    m1, m2 = map(lambda img: (img.astype(np.float) - img.min()) / (img.max() - img.min()), (m1, m2))
    cm1 = m1 * 255
    cm2 = m2 * 255

    fast = cv2.FastFeatureDetector_create(nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)
    kp1 = fast.detect(np.uint8(cm1), None)
    kp2 = fast.detect(np.uint8(cm2), None)

    m1_point = kp2point(kp1)
    m2_point = kp2point(kp2)
    # print(m1_point.shape)

    mim1, _, _, _ = RIFT(psd_img_1, m1_point, eo1, 96, 4, 6)
    mim2, _, _, _ = RIFT(psd_img_2, m2_point, eo2, 96, 4, 6)

    # print(mim1, mim2)

    mim_img1 = np.zeros([H, W, 3], int)
    mim_img2 = np.zeros([H, W, 3], int)
    for i in range(3):
        mim_img1[..., i] = mim1
    for i in range(3):
        mim_img2[..., i] = mim2
    result1 = np.zeros(mim1.shape, dtype=np.float32)
    cv2.normalize(mim1, result1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    result2 = np.zeros(mim2.shape, dtype=np.float32)
    cv2.normalize(mim2, result2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # print(mim_img1)
    # print(result1)
    mim_path1 = out_dir + '/' + file_name + '_hwmim.jpg'
    mim_path2 = out_dir + '/' + file_name + '_kjgmim.jpg'

    cv2.imwrite(mim_path1, np.uint8(result1 * 255.0))
    cv2.imwrite(mim_path2, np.uint8(result2 * 255.0))

    im11 = mim_path1
    im22 = mim_path2
    psd_img_11 = cv2.imread(im11, cv2.IMREAD_COLOR)
    psd_img_22 = cv2.imread(im22, cv2.IMREAD_COLOR)
    psd_img_11 = cv2.resize(psd_img_11, (W, H))
    psd_img_22 = cv2.resize(psd_img_22, (W, H))

    image0, inp0, scales0 = read_image(
        im11, device, [W, H], 0, None)
    image1, inp1, scales1 = read_image(
        im22, device, [W, H], 0, None)
    # baseline
    batch = {'image0': inp0, 'image1': inp1}
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    color = cm.jet(mconf, alpha=0.7)
    print(mconf.shape, color.shape)
    text = []

    out = make_matching_plot_color_Loftr(mconf, show_THR,
        psd_img_11, psd_img_22, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=False, small_text=[], WW=W, HH=H)
    out_file = out_dir + '/' + file_name + '_loftrmatchmim.jpg'
    cv2.imwrite(out_file, out)
    print(out_file)
    (HH, status) = cv2.findHomography(mkpts0, mkpts1, cv2.RHO, ransacReprojThreshold=5.0)
    result = cv2.warpPerspective(psd_img_1, HH, (max(psd_img_1.shape[1], psd_img_2.shape[1]), psd_img_2.shape[0]))
    fused = psd_img_2 // 2 + result // 2
    img_sg = fused
    out_file = out_dir + '/' + file_name + '_mergemim.jpg'
    # print('\nWriting image to {}'.format(out_file))
    cv2.imwrite(out_file, img_sg)
    print(out_file)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(psd_img_1, None)  # des是描述子
    kp2, des2 = sift.detectAndCompute(psd_img_2, None)  # des是描述子
    bf = cv2.BFMatcher()
    # print(des1, des2)
    matches = bf.knnMatch(des1, des2, k=2)
    # 调整ratio
    good = []
    dist = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good.append([m])
            dist.append(m.distance)
    num_match = len(good)
    img_match = cv2.drawMatchesKnn(psd_img_11, kp1, psd_img_22, kp2, good, None, flags=2)
    out_file = out_dir + '/' + file_name + '_matchsift.jpg'
    cv2.imwrite(out_file, img_match)
    print(out_file)



