
import cv2
import time
import numpy as np
import matplotlib.cm as cm
import torch
import random
import math

def FSC(cor1, cor2, change_form, error_t):
    (M, N) = np.shape(cor1)
    if (change_form == 'similarity'):
        n = 2
        max_iteration = M * (M - 1) / 2
    elif (change_form == 'affine'):
        n = 3
        max_iteration = M * (M - 1) * (M - 2) / (2 * 3)
    elif (change_form == 'perspective'):
        n = 4
        max_iteration = M * (M - 1) * (M - 2) / (2 * 3)

    if (max_iteration > 10000):
        iterations = 10000
    else:
        iterations = max_iteration

    most_consensus_number = 0
    cor1_new = np.zeros([M, N])
    cor2_new = np.zeros([M, N])

    for i in range(iterations):
        while (True):
            a = np.floor(1 + (M - 1) * np.random.rand(1, n)).astype(np.int)[0]
            cor11 = cor1[a]
            cor22 = cor2[a]
            if n == 2 and (a[0] != a[1]) and sum(cor11[0] != cor11[1]) and sum(cor22[0] != cor22[1]):
                break
            if n == 3 and (a[0] != a[1] and a[0] != a[2] and a[1] != a[2]) and sum(cor11[0] != cor11[1]) and sum(
                    cor11[0] != cor11[2]) and sum(cor11[1] != cor11[2]) and sum(cor22[0] != cor22[1]) and sum(
                cor22[0] != cor22[2]) and sum(cor22[1] != cor22[2]):
                break
            if n == 4 and (
                    a[0] != a[1] and a[0] != a[2] and a[0] != a[3] and a[1] != a[2] and a[1] != a[3] and a[2] != a[
                3]) and sum(cor11[0] != cor11[1]) and sum(cor11[0] != cor11[2]) and sum(cor11[0] != cor11[3]) and sum(
                cor11[1] != cor11[2]) and sum(cor11[1] != cor11[3]) and sum(cor11[2] != cor11[3]) and sum(
                cor22[0] != cor11[1]) and sum(cor22[0] != cor22[2]) and sum(cor22[0] != cor22[3]) and sum(
                cor22[1] != cor22[2]) and sum(cor22[1] != cor22[3]) and sum(cor22[2] != cor22[3]):
                break
        parameters, __ = LSM(cor11, cor22, change_form)
        solution = np.array([[parameters[0], parameters[1], parameters[4]],
                             [parameters[2], parameters[3], parameters[5]],
                             [parameters[6], parameters[7], 1]])
        match1_xy = np.ones([3, len(cor1)])
        match1_xy[:2] = cor1.T

        if change_form == 'affine':
            t_match1_xy = solution.dot(match1_xy)
            match2_xy = np.ones([3, len(cor1)])
            match2_xy[:2] = cor2.T
            diff_match2_xy = t_match1_xy - match2_xy
            diff_match2_xy = np.sqrt(sum(np.power(diff_match2_xy, 2)))
            index_in = np.argwhere(diff_match2_xy < error_t)
            consensus_num = len(index_in)
            index_in = np.squeeze(index_in)

        if consensus_num > most_consensus_number:
            most_consensus_number = consensus_num
            cor1_new = cor1[index_in]
            cor2_new = cor2[index_in]
    unil = cor1_new
    __, IA = np.unique(unil, return_index=True, axis=0)
    IA_new = np.sort(IA)
    cor1_new = cor1_new[IA_new]
    cor2_new = cor2_new[IA_new]
    unil = cor2_new
    __, IA = np.unique(unil, return_index=True, axis=0)
    IA_new = np.sort(IA)
    cor1_new = cor1_new[IA_new]
    cor2_new = cor2_new[IA_new]

    parameters, rmse = LSM(cor1_new, cor2_new, change_form)
    solution = np.array([[parameters[0], parameters[1], parameters[4]],
                         [parameters[2], parameters[3], parameters[5]],
                         [parameters[6], parameters[7], 1]])
    return solution

def LSM(match1, match2, change_form):
    A = np.zeros([2 * len(match1), 4])
    for i in range(len(match1)):
        A[2 * i:2 * i + 2] = np.tile(match1[i], (2, 2))
    B = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    B = np.tile(B, (len(match1), 1))
    A = A * B
    B = np.array([[1, 0], [0, 1]])
    B = np.tile(B, (len(match1), 1))
    A = np.hstack((A, B))
    b = match2.reshape(1, int(len(match2) * len(match2[0]))).T

    if change_form == "affine":
        Q, R = np.linalg.qr(A)
        parameters = np.zeros([8, 1])
        parameters[:6] = np.linalg.solve(R, np.dot(Q.T, b))
        N = len(match1)
        M = np.array([[parameters[0][0], parameters[1][0]], [parameters[2][0], parameters[3][0]]])
        match1_test_trans = M.dot(match1.T) + np.tile([parameters[4], parameters[5]], (1, N))
        match1_test_trans = match1_test_trans.T
        test = match1_test_trans - match2
        rmse = math.sqrt(sum(sum(np.power(test, 2))) / N)
    return np.squeeze(parameters), rmse


def RIFT(img, kps, eo, patch_size=96, s=4, o=6):
    KPS = kps.T
    (yim, xim, _) = np.shape(img)
    CS = np.zeros([yim, xim, o], np.float)
    MIM = np.zeros([yim, xim], np.float)
    for j in range(o):
        for i in range(s):
            # 将各个scale的变换结果的幅度相加
            CS[..., j] = CS[..., j] + np.abs(np.array(eo[j][i]))

    mim = np.argmax(CS, axis=2)
    des = np.zeros([36 * o, np.size(KPS, 1)])
    kps_to_ignore = np.ones([1, np.size(KPS, 1)], bool)
    for k in range(np.size(KPS, 1)):
        x = round(KPS[0][k])
        y = round(KPS[1][k])
        x1 = max(0, x - math.floor(patch_size / 2))
        y1 = max(0, y - math.floor(patch_size / 2))
        x2 = min(x + math.floor(patch_size / 2), np.size(img, 1))
        y2 = min(y + math.floor(patch_size / 2), np.size(img, 0))

        if y2 - y1 != patch_size or x2 - x1 != patch_size:
            kps_to_ignore[0][i] = 0
            continue
    for x in range(xim):
        for y in range(yim):
            MIM[y, x] = CS[y, x, mim[y, x]]
    # return MIM, CS
    # print(mim.shape)

    patch = mim[y1:y2, x1:x2]
    # print(patch.shape)
    ys, xs = np.size(patch, 0), np.size(patch, 1)
    ns = 6
    RIFT_des = np.zeros([ns, ns, o])
    for j in range(ns):
        for i in range(ns):
            clip = patch[round((j) * ys / ns):round((j + 1) * ys / ns),
                   round((i) * xs / ns): round((i + 1) * xs / ns)]
            x, __ = np.histogram(clip.T.flatten(), bins=6, range=(0, o), density=False)
            te = RIFT_des[j][i]
            RIFT_des[j][i] = x.reshape(1, 1, len(x))
    RIFT_des = RIFT_des.T.flatten()

    df = np.linalg.norm(RIFT_des)
    if df != 0:
        RIFT_des = RIFT_des / df
    des[:, [k]] = np.expand_dims(RIFT_des, axis=1)
    # print(kps_to_ignore.shape)
    # print(KPS.shape)
    # print(len(des))
    # m = repeat(kps_to_ignore, '1 n -> c n', c=2)
    m = np.repeat(kps_to_ignore, 2, 0)
    # print(m.shape)
    v = KPS[m]
    KPS_out = v.reshape(2, int(len(v) / 2)).T
    # w = repeat(kps_to_ignore, '1 n -> c n', c=len(des))
    w = np.repeat(kps_to_ignore, len(des), 0)
    z = des[w]
    des_out = z.reshape(len(des), int(len(z) / len(des))).T
    des_out = np.float32(des_out) * 100
    # return KPS_out, des_out
    # print(KPS_out.shape, des_out.shape)
    return MIM, CS, KPS_out, des_out
