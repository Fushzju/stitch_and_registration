import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp
out_dir = osp.join('output_data', 'reg_out')

num_match_base = np.load(out_dir + '/data_stat/num_match_base.npy')
num_match_rift = np.load(out_dir + '/data_stat/num_match_rift.npy')
num_match_gan = np.load(out_dir + '/data_stat/num_match_gan.npy')
# num_match_ormim = np.load('num_match_ormim.npy')
# num_match_sift = np.load('num_match_sift.npy')
# MI_base = np.load('MI_base.npy')
# MI_rift = np.load('MI_rift.npy')

# rate_base = np.load('rate_base.npy')
# rate_rift = np.load('rate_rift.npy')

x_axis_data = [i for i in range(300)]
y_axis_data1 = num_match_base
y_axis_data2 = num_match_rift
y_axis_data3 = num_match_gan
y_axis_data1.sort()
y_axis_data2.sort()
y_axis_data3.sort()
plt.plot(x_axis_data, y_axis_data1, label='Original Inputs', color='b')
plt.plot(x_axis_data, y_axis_data2, label='With modal-invariant maps', color='r')
plt.plot(x_axis_data, y_axis_data3, label='CycleGAN transformation', color='k')
plt.ylabel('Match num')
plt.xlabel('Image No.')
plt.legend(loc='best')
plt.savefig('result2/img/MI_match1.jpg')


# plt.plot(x_axis_data, y_axis_data1, label='Original Inputs', color='b')
# plt.plot(x_axis_data, y_axis_data2, label='With modal-invariant maps', color='r')
#
# plt.ylabel('Mutual Information')
# plt.xlabel('Image No.')
# plt.legend(loc='best')
# plt.savefig('result2/img/MI_match1.jpg')