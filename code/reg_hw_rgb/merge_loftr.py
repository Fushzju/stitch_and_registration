import cv2
import time
import numpy as np
import matplotlib.cm as cm
import torch
import random
from models.matching import Matching
from scipy import optimize
import os
import os.path as osp
from glob import glob
from basemerge import merge_sift, merge_sg, merge_loftr
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
    working_dir = '/home/fsh/stitching_and_registration'
    # data_path = 'data_hw_rgb'
    # out_path = 'result'
    data_path = 'flir'
    out_path = 'result3'
    data_dir = osp.join(working_dir, 'input_data')
    data_dir = osp.join(data_dir, 'reg_data')
    images_dir = osp.join(data_dir, data_path)
    out_dir = osp.join(osp.join(working_dir, 'output_data'), 'reg_out')
    # out_dir = osp.join(out_dir, out_path)

    path = 'RGBInputImage'

    fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
    j = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    print(images_dir)
    num_match_base = []
    num_match_rift = []
    num_match_gan = []
    num_match_ormim = []
    num_match_sift = []
    MI_base = []
    MI_rift = []
    MI_gan = []


    print(len(fpaths))
    for file in fpaths:
        j = j + 1
        print(j)
        # if j < 200:
        #     continue
        if j > 3:
            break
        # print(file)
        path_current = osp.dirname(osp.abspath(__file__))
        im_root, im_file = get_path(file)
        if data_path == 'data_hw_rgb':
            im1 = images_dir + '/RawInputImage/' + im_file
        elif data_path == 'flir':
            im1 = images_dir + '/RawInputImage/' + im_file[-25:-8] + '_hw.jpg'
        im2 = file
        # print(im1, im2)
        if not osp.exists(im1):
            continue

        H = 360
        W = 480

        psd_img_1 = cv2.imread(im1, cv2.IMREAD_COLOR)
        psd_img_2 = cv2.imread(im2, cv2.IMREAD_COLOR)
        psd_img_1 = cv2.resize(psd_img_1, (W, H))
        psd_img_2 = cv2.resize(psd_img_2, (W, H))

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
        mim_path1 = out_dir + '/' + out_path + '/Loftr_mimout/MIMout/IFMIM/'
        mim_path2 = out_dir + '/' + out_path + '/Loftr_mimout/MIMout/RGBMIM/'
        if not os.path.exists(mim_path1):
            os.makedirs(mim_path1)
        if not os.path.exists(mim_path2):
            os.makedirs(mim_path2)
        mim_path1 = mim_path1 + im_file
        mim_path2 = mim_path2 + im_file
        cv2.imwrite(mim_path1, np.uint8(result1 * 255.0))
        cv2.imwrite(mim_path2, np.uint8(result2 * 255.0))

        netG_A2B = Generator(3, 3)
        netG_B2A = Generator(3, 3)

        # if opt.cuda:
        netG_A2B.to(device)
        netG_B2A.to(device)

        # Load state dicts
        GAN_path1 = working_dir + '/code/reg_hw_rgb/gan/netG_A2B.pth'
        GAN_path2 = working_dir + '/code/reg_hw_rgb/gan/netG_B2A.pth'
        netG_A2B.load_state_dict(torch.load(GAN_path1))
        netG_B2A.load_state_dict(torch.load(GAN_path2))

        # Set model's test mode
        netG_A2B.eval()
        netG_B2A.eval()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor

        # Dataset loader
        transforms_ = transforms.Compose([transforms.Resize((360, 480)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Set model input
        real_A = transforms_(Image.open(im1)).unsqueeze(0).cuda()
        real_B = transforms_(Image.open(im2)).unsqueeze(0).cuda()
        # print(real_A.size())
        # Generate output
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        # Save image files
        gan_dir = out_dir + '/' + out_path + '/Loftr_ganout/GAN_Fake_RGB/'
        if not os.path.exists(gan_dir):
            os.makedirs(gan_dir)
        gan_path = gan_dir + im_file

        save_image(fake_B, gan_path)
        mode_path1 = "Loftr_baseout"
        mode_path2 = "Loftr_mimout"
        mode_path3 = "Loftr_ganout"
        mode_path4 = "orgmimout"
        mode_path5 = "siftout"

        num_base, mi_base = merge_loftr(mode_path1, out_path, im1, im2, device, W, H, psd_img_1, psd_img_2,
                                              out_dir, im_file, matcher)
        num_mim, mi_rift = merge_loftr(mode_path2, out_path, mim_path1, mim_path2, device, W, H, psd_img_1,
                                             psd_img_2, out_dir, im_file, matcher)
        num_gan, mi_gan = merge_loftr(mode_path3, out_path, gan_path, im2, device, W, H, psd_img_1, psd_img_2,
                                           out_dir, im_file, matcher)
        num_orift = merge_sift(mode_path4, out_path, mim_path1, mim_path2, device, W, H, psd_img_1,
                                                 psd_img_2, out_dir, im_file, matcher)
        num_sift = merge_sift(mode_path5, out_path, mim_path1, mim_path2, device, W, H, psd_img_1,
                                                 psd_img_2, out_dir, im_file, matcher)
        # rate_base.append(r_base)
        # rate_rift.append(r_rift)
        # rate_gan.append(r_gan)
        # rate_ormim.append(r_orift)
        print(mi_base, mi_rift, mi_gan)
        print(num_base, num_mim, num_gan, num_orift, num_sift)
        num_match_base.append(num_base)
        num_match_rift.append(num_mim)
        num_match_gan.append(num_gan)
        num_match_ormim.append(num_orift)
        num_match_sift.append(num_sift)

        MI_base.append(mi_base)
        MI_rift.append(mi_rift)
        MI_gan.append(mi_gan)
        # MI_ormim.append(mi_orift)

    print(num_match_base, num_match_rift, num_match_gan, num_match_ormim, num_match_sift)
    num_match_base = np.array(num_match_base)
    num_match_rift = np.array(num_match_rift)
    num_match_gan = np.array(num_match_gan)
    num_match_ormim = np.array(num_match_ormim)
    num_match_sift = np.array(num_match_sift)
    print(MI_base, MI_rift, MI_gan)
    MI_base = np.array(MI_base)
    MI_rift = np.array(MI_rift)
    MI_gan = np.array(MI_gan)
    # MI_ormim = np.array(MI_ormim)

    # rate_base = np.array(rate_base)
    # rate_rift = np.array(rate_rift)
    # rate_gan = np.array(rate_gan)
    # rate_ormim = np.array(rate_ormim)

    data_stat_dir = osp.join(osp.join(out_dir, out_path), 'data_stat')
    if not os.path.exists(data_stat_dir):
        os.makedirs(data_stat_dir)

    np.save(data_stat_dir + '/num_match_loftr_base.npy', num_match_base)
    np.save(data_stat_dir + '/num_match_loftr_rift.npy', num_match_rift)
    np.save(data_stat_dir + '/num_match_loftr_gan.npy', num_match_gan)
    np.save(data_stat_dir + '/num_match_ormim.npy', num_match_ormim)
    np.save(data_stat_dir + '/num_match_sift.npy', num_match_sift)

    np.save(data_stat_dir + '/MI_loftr_base.npy', MI_base)
    np.save(data_stat_dir + '/MI_loftr_rift.npy', MI_rift)
    np.save(data_stat_dir + '/MI_loftr_gan.npy', MI_gan)
    # np.save('MI_ormim.npy', MI_ormim)

    # np.save('rate_base.npy', rate_base)
    # np.save('rate_rift.npy', rate_rift)
    # np.save('rate_gan.npy', rate_gan)
    # np.save('rate_ormim.npy', rate_ormim)

    print(MI_base.mean(), MI_rift.mean(), MI_gan.mean())
    # print(rate_base.mean(), rate_rift.mean(), rate_gan.mean(), rate_ormim.mean())
    print(num_match_base.mean(), num_match_rift.mean(), num_match_gan.mean(), num_match_ormim.mean(), num_match_sift.mean())
