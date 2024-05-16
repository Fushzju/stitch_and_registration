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
from basemerge import basemerge, basemerge2, basemerge3
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

import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger
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
    # out_path = 'result'
    out_path = 'result4'


    path = 'examples'
    image_dir = osp.join(working_dir, path)
    fpaths = sorted(glob(osp.join(working_dir, path, '*')))
    j = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    num_match_base = []
    num_match_rift = []
    num_match_ormim = []
    num_match_sift = []
    MI_base = []
    MI_rift = []
    MI_gan = []

    print(fpaths)

    for file in fpaths:
        j = j + 1
        print(j)
        # if j < 200:
        #     continue
        print(file)
        # path_current = osp.dirname(osp.abspath(__file__))
        # im_root, im_file = get_path(file)
        fpathss = os.listdir(file)
        # print(fpathss)
        file_name = fpathss[0][:17]
        print(file_name)
        im1 = file + '/' + file_name + '_hw.jpg'
        im2 = file + '/' + file_name + '_kjg.jpg'
        im11 = file + '/' + file_name + '_hwmim.jpg'
        im22 = file + '/' + file_name + '_kjgmim.jpg'
        print(im1, im2, im11, im22)
        if not osp.exists(im1):
            continue

        H = 360
        W = 480

        psd_img_1 = cv2.imread(im1, cv2.IMREAD_COLOR)
        psd_img_2 = cv2.imread(im2, cv2.IMREAD_COLOR)
        psd_img_1 = cv2.resize(psd_img_1, (W, H))
        psd_img_2 = cv2.resize(psd_img_2, (W, H))

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
        show_THR = 0.3
        out = make_matching_plot_color_Loftr(mconf, show_THR,
            psd_img_11, psd_img_22, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=False, small_text=[], WW=W, HH=H)
        out_file = file + '/' + file_name + '_loftrmatchmim.jpg'
        cv2.imwrite(out_file, out)
        print(out_file)
        (HH, status) = cv2.findHomography(mkpts0, mkpts1, cv2.RHO, ransacReprojThreshold=5.0)
        result = cv2.warpPerspective(psd_img_1, HH, (max(psd_img_1.shape[1], psd_img_2.shape[1]), psd_img_2.shape[0]))
        fused = psd_img_2 // 2 + result // 2
        img_sg = fused
        out_file = file + '/' + file_name + '_mergemim.jpg'
        # print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, img_sg)
        print(out_file)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(psd_img_1, None)  # des是描述子
        kp2, des2 = sift.detectAndCompute(psd_img_2, None)  # des是描述子
        bf = cv2.BFMatcher()
        # print(des1, des2)
        if des1 is None or des2 is None:
            continue
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
        out_file = file + '/' + file_name + '_matchsift.jpg'
        cv2.imwrite(out_file, img_match)
        print(out_file)


