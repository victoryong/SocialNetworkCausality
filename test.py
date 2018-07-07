import utils.config_util as conf

import numpy as np
import csv
import os

# a = np.array([[[1,0,0],[0,1,2]], [[0,0,0],[2,2,2]], [[7,8,9],[10,11,12]]])
# print(np.concatenate((a[0], a[1]), axis=1))
# print(np.concatenate((np.array([]), a[0])))

# Read seq and count numbers
# seq = []
# with open(conf.get_filename_via_tpl('seq', n_users=10, n_samples=2192)) as fp:
#     csv_reader = csv.reader(fp)
#     for line in csv_reader:
#         print(sum([int(x) for x in line]))
#         seq.append([int(x) for x in line])
#     print('--')
#
#     aaa = []
#     for i in range(10):
#         aa = []
#         for j in range(10):
#             a = 0
#             for k in range(2191):
#                 if seq[i][k] == 0 and seq[j][k+1] == 0:
#                     a += 1
#             aa.append(a)
#         aaa.append(aa)
#     print(np.array(aaa))

# Read lda data and show some lines
# from main import load_vec
# lda = load_vec('lsi', 100)
# count = 0
# print(lda[0][3:6])
# for i in lda[0]:
#     if np.sum(i, axis=0) == 0.:
#         count += 1
# print(count)

# import matplotlib.pyplot as plt
# seq = np.loadtxt(conf.get_filename_via_tpl('seq', n_users=10, n_samples=2192), delimiter=',')
# print(seq)
# print(seq.shape)
#
# hist = seq.sum(0)
# # print(hist.shape)
# x = np.array(range(1, 2193))
# # hist = np.array(hist).reshape((2192, 1))
# # histogram = np.histogram2d(x, hist)
#
# plt.bar(x, hist, 1.)
# plt.show()2047790442
import sys
csv.field_size_limit(sys.maxsize)
with open(conf.get_filename_via_tpl('text', user_id=2263978304, n_samples=626)) as fp:
    # lines = fp.readlines()
    # print(len(lines))
    csv_reader = csv.reader(fp)
    idx = 0
    t = []
    for i in csv_reader:
        if len(i) and i[0] == '':
            t.append([])
            print([])
        else:
            # t.append(i)
            print(i)
        idx += 1
        if idx > 410:
            break

    for j in range(419, 423):
        print(j, len(lines[j]))
# print(idx, csv.field_size_limit())
