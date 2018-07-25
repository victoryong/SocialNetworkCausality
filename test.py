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
# np.set_printoptions(threshold=np.inf)
# original_seq = np.loadtxt(conf.get_filename_via_tpl('seq', n_users=12, n_samples=2192), delimiter=',')
# print(original_seq.shape)
#
# for i in original_seq:
#     print(i)
#
# # from model_object import make_seq_via_time_steps
# # s = make_seq_via_time_steps(original_seq, [2,4,8,2192])
# # print(s)
# print(original_seq[:, 0])
# print(original_seq[:, 1])
# print(original_seq[:, 0] | original_seq[:, 1])
# original_seq[:, 0] = original_seq[:, 0] | original_seq[:, 1]
# print(original_seq[:, 0])

from matplotlib import pyplot as plt
res = []
with open('search_timesteps_results_inner_pro.csv', 'r', newline='') as fp:
    csv_reader = csv.reader(fp)
    for i in csv_reader:
        res.append(i)

res = np.array(res[1:])
objs = res[:, 1]

lens = res[:, -1]
lens = np.sort(lens)

hseqs= res[:, 2]
its = res[:, 3]


plt.plot(lens, objs)

plt.show()

print(objs)
print(lens)
print(hseqs)


