import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp

out_dir = osp.join('output_data', 'reg_out')

# rate_base = np.load('rate_base.npy')
# rate_rift = np.load('rate_rift.npy')
# rate_gan = np.load('rate_gan.npy')

num_match_base = np.load(out_dir + '/data_stat/num_match_loftr_base.npy')
num_match_rift = np.load(out_dir + '/data_stat/num_match_loftr_rift.npy')
num_match_gan = np.load(out_dir + '/data_stat/num_match_loftr_gan.npy')
num_match_ormim = np.load(out_dir + '/data_stat/num_match_ormim.npy')
num_match_sift = np.load(out_dir + '/data_stat/num_match_sift.npy')
MI_base = np.load(out_dir + '/data_stat/MI_base.npy')
MI_rift = np.load(out_dir + '/data_stat/MI_rift.npy')

# rate_base = np.load('rate_base.npy')
# rate_rift = np.load('rate_rift.npy')

x_axis_data = [i for i in range(300)]
y_axis_data1 = MI_base
y_axis_data2 = MI_rift
y_axis_data1.sort()
y_axis_data2.sort()
plt.plot(x_axis_data, y_axis_data1, label='Original Inputs', color='b')
plt.plot(x_axis_data, y_axis_data2, label='With modal-invariant maps', color='r')

plt.ylabel('Mutual Information')
# plt.ylabel('Repeatability')
# plt.ylabel('Mutual Information')
plt.xlabel('Image No.')
# plt.plot(x_axis_data, y_axis_data3, label='gan', color='b')
plt.legend(loc='best')
# plt.savefig('result2/MI.jpg')
plt.savefig('result3/imgs/MI_match1.jpg')
# y_axis_data3 = MI_gan
# y_axis_data3 = num_match_gan
# y_axis_data1 = rate_base
# y_axis_data2 = rate_rift
# print(y_axis_data2)
y_axis_data1 = num_match_base
y_axis_data2 = num_match_rift
y_axis_data3 = num_match_sift
y_axis_data4 = num_match_ormim

y_axis_data1.sort()
y_axis_data2.sort()
y_axis_data3.sort()
y_axis_data4.sort()
plt.rcParams['figure.figsize'] = (8, 4)  # 单位是inches
# plt.plot(x_axis_data, y_axis_data1, label='LoFTR without radiation-invariant transform', color='b')
# plt.plot(x_axis_data, y_axis_data2, label='LoFTR with radiation-invariant transform', color='r')
# plt.plot(x_axis_data, y_axis_data3, label='SIFT without radiation-invariant transform', color='k')
# plt.plot(x_axis_data, y_axis_data4, label='SIFT with radiation-invariant transform', color='g')
# plt.plot(x_axis_data, y_axis_data1, label='SIFT', color='b')
# plt.plot(x_axis_data, y_axis_data2, label='LoFTR', color='r')
xx1 = ['<300', '301-600', '601-900', '901-1200', '>1200']
xx2 = ['<50', '51-100', '101-150', '151-200', '>200']
xx = xx1
bins1 = [0, 300, 600, 900, 1200, 2000]
bins2 = [0, 50, 100, 150, 200, 2000]
yy1 = pd.value_counts(pd.cut(y_axis_data1, bins1), sort=False)
yy2 = pd.value_counts(pd.cut(y_axis_data2, bins1), sort=False)
yy3 = pd.value_counts(pd.cut(y_axis_data3, bins1), sort=False)
yy4 = pd.value_counts(pd.cut(y_axis_data4, bins2), sort=False)
print(yy2, yy1)

width_bar = 0.2

x_1 = [i + 1*width_bar for i in range(len(xx))]

# 这里的目的是把条形图间隔开来，防止重叠，都加上width_bar的倍数
x_2 = [i for i in range(len(xx))]

x_3 = [i + 0.5*width_bar for i in range(len(xx))]

# 设置图片大小和分辨率，其中figure单位为英寸，dpi为分辨率
plt.figure(figsize=(6,4))

# 设置X轴的刻度和字符串，步长为1
plt.xticks(x_3, xx)

# plt.bar()用于画条形图


plt.title("LoFTR Matching")
# plt.title("Radiation-invariant Matching")

# c_1 = plt.bar(x_1, yy4, width=width_bar, label="SIFT")
# c_2 = plt.bar(x_2, yy2, width=width_bar, label="LoFTR")

c_1 = plt.bar(x_1, yy1, width=width_bar, label="Original inputs")
c_2 = plt.bar(x_2, yy2, width=width_bar, label="With modal-invariant maps")

# 用于设置数字标注
# for k in c_1:
#     height = k.get_height()
#     # print(k.get_x(), k.get_width())
#     plt.text(k.get_x() + k.get_width(), height, None, fontsize=10, ha="center", va="bottom")
#
# for k in c_2:
#     height = k.get_height()
#     # print(k.get_x(), k.get_width())
#     plt.text(k.get_x() + k.get_width(), height, None, fontsize=10, ha="center", va="bottom")

plt.xlabel('Match num')
# plt.ylabel('Repeatability')
# plt.ylabel('Mutual Information')
plt.ylabel('Image pair num')
# plt.plot(x_axis_data, y_axis_data3, label='gan', color='b')
plt.legend(loc='best')
# plt.savefig('result2/MI.jpg')
plt.savefig('result3/imgs/bar_match1.jpg')
print(y_axis_data2.mean(), y_axis_data4.mean())
