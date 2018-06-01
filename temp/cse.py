# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 21:13:40 2014

@author: yuanchang
"""

import numpy as np
from pandas import Series
from utils.te import cmidd
import os,time,random
from copy import deepcopy
import pandas as pd

import utils.config_util as conf

os.system('cls')
lagmax = 6
tmax = conf.N_SAMPLES
nnode = conf.N_USERS
resampleTime = 100

length = tmax - lagmax-1
lag_max_pre = np.eye(nnode, dtype=int)
lag_max_late = np.zeros((nnode,nnode),dtype=np.int)
lag_te = np.zeros((nnode,nnode))
sample = np.loadtxt(conf.get_data_filename_via_template('seq', n_users=nnode, n_samples=conf.N_SAMPLES), delimiter=',')
# sample = sample.T
t = time.time()
for n in range(nnode):#xrange(0):
    con = {}
    con[(n,1)] = sample[n,lagmax:tmax-1]
    x = sample[n,lagmax+1:tmax]
    redundance = 0
    nodeset = range(nnode)

    for j in range(lagmax):
        n_teList = []

        for i in nodeset:
            if n==i:
                temp_te = cmidd(x,sample[i,(lagmax-j-1):(tmax-j-2)],con)
                temp_te_2 = np.array([cmidd(x,random.sample(list(sample[i,(lagmax-j-1):(tmax-j-2)]),length),con) for m in range(resampleTime)])
                if len(temp_te_2[temp_te_2 > temp_te])/100.00 < 0.01:
                    n_teList.append(i)
                    con[(i,j+2)] = sample[i,(lagmax-j-1):(tmax-j-2)]
            else:
#                        y_next = sample[i,(lagmax-j+1):(tmax-j)]
                temp_te = cmidd(x,sample[i,(lagmax-j+1):(tmax-j)],con)
                temp_te_2 = np.array([cmidd(x,random.sample(list(sample[i,lagmax-j+1:tmax-j]),length),con) for m in range(resampleTime)])
                if len(temp_te_2[temp_te_2 > temp_te])/100.00 < 0.01:
                    n_teList.append(i)
                    con[(i,j+1)] = sample[i,(lagmax-j+1):(tmax-j)]

        lag_max_pre[n,n_teList] += 1

        if len(n_teList):
            nodeset = n_teList[:]
        else:
            break
    nodeindex = lag_max_pre[n].nonzero()[0]
    for i in nodeindex:
        tem_con = deepcopy(con)
        j = lag_max_pre[n,i]
        while (j > 0) and bool(len(con)):
                tem_con = deepcopy(con)
                y_next = tem_con.pop((i,j))
                temp_te = cmidd(x,y_next,tem_con)
                temp_te_2 = np.array([cmidd(x,random.sample(list(y_next),length),tem_con) for m in range(resampleTime)])
                if len(temp_te_2[temp_te_2 > temp_te])/100.00 < 0.01:
                    break
                else:
                    con = tem_con
                    j -= 1
        if j and len(con):
            for l in range(1,j+1):
                tem_con = deepcopy(con)
                lag_te[n,i] += cmidd(x,tem_con.pop((i,l)),tem_con)
            lag_max_late[n,i] = j


for n in range(nnode):
    for i in range(nnode):
        if n != i :
            if (lag_te[n,i]>=lag_te[i,n]):
                lag_max_late[i,n] = 0
            elif lag_te[i,n]:
                lag_max_late[n,i] = 0
        else:
            break

t2= time.time() - t
np.savetxt(conf.get_data_filename_via_template(
    're_lag_max_pre', postfix='txt', n_users=nnode, n_samples=tmax), lag_max_pre,fmt='%d',delimiter=',')
np.savetxt(conf.get_data_filename_via_template(
    're_lag_max_late', postfix='txt', n_users=nnode, n_samples=tmax),lag_max_late,fmt='%d',delimiter=',')
np.savetxt(conf.get_data_filename_via_template(
    're_lag_te', postfix='txt', n_users=nnode, n_samples=tmax),lag_te,delimiter=',')

