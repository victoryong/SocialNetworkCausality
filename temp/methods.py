# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 21:13:40 2014

@author: yuanchang
"""
import random
from copy import deepcopy

import numpy as np

import utils.config_util as conf
from temp.te import cmidd
from utils.log import get_console_logger

logger = get_console_logger(__name__)


def testMNR(lagmax,tmax,resampleTime,nnode,sample,indegreeaverage):
    """
    function of MNR
    """
    length = tmax - lagmax-1
    lag_max_pre = np.eye(nnode, dtype=int)
    lag_max_late = np.zeros((nnode,nnode),dtype=np.int)
    lag_te = np.zeros((nnode,nnode))
    # sample = sample.T
    for n in range(nnode):
        con = {}
        con[(n,1)] = sample[n,lagmax:tmax-1]
        x = sample[n,lagmax+1:tmax]
        nodeset = range(nnode)

        ####find lag&pc####
        for j in range(lagmax):
            n_teList = []
            for i in nodeset:
                if n==i:
                    temp_te = cmidd(x,sample[i,(lagmax-j-1):(tmax-j-2)],con)
                    temp_te_2 = np.array([cmidd(x,random.sample(sample[i,(lagmax-j-1):(tmax-j-2)].tolist(),length),con) for m in range(resampleTime)])
                    if len(temp_te_2[temp_te_2 > temp_te])/100.00 < 0.01:
                        n_teList.append(i)
                        con[(i,j+2)] = sample[i,(lagmax-j-1):(tmax-j-2)]
                else:
                    temp_te = cmidd(x,sample[i,(lagmax-j+1):(tmax-j)],con)
                    temp_te_2 = np.array([cmidd(x,random.sample(sample[i,(lagmax-j+1):(tmax-j)].tolist(),length),con) for m in range(resampleTime)])
                    if len(temp_te_2[temp_te_2 > temp_te])/100.00 < 0.01:
                        n_teList.append(i)
                        con[(i,j+1)] = sample[i,(lagmax-j+1):(tmax-j)]

            lag_max_pre[n,n_teList] += 1

            if len(n_teList):
                nodeset = n_teList[:]
            else:
                break

        ####remove lag&pc####
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
    np.savetxt(conf.get_filename_via_tpl(
        'mnr_lag_max_pre', n_users=nnode, n_samples=tmax), lag_max_pre, fmt='%d', delimiter=',')
    np.savetxt(conf.get_filename_via_tpl(
        'mnr_lag_max_late', n_users=nnode, n_samples=tmax), lag_max_late, fmt='%d', delimiter=',')
    np.savetxt(conf.get_filename_via_tpl(
        'mnr_lag_te', n_users=nnode, n_samples=tmax), lag_te, delimiter=',')
    logger.info('MNR results have been saved in result folder.')
    return lag_max_pre,lag_max_late,lag_te

def testCSE(lagmax,tmax,resampleTime,nnode,sample,indegreeaverage):
    """
    function of CSE
    """
    length = tmax - lagmax-1
    lag_max_pre = np.eye(nnode, dtype=int)
    lag_max_late = np.zeros((nnode,nnode),dtype=np.int)
    lag_te = np.zeros((nnode,nnode))
    # sample = sample.T
    for n in range(nnode):
        con = {}
        con[(n,1)] = sample[n,lagmax:tmax-1]
        x = sample[n,lagmax+1:tmax]
        nodeset = range(nnode)
        n_teList = []

        ####find lag&pc####
        for i in nodeset:
            if n==i:
                temp_te = cmidd(x,sample[i,(lagmax-1):(tmax-2)],con)
                temp_te_2 = np.array([cmidd(x,random.sample(sample[i,(lagmax-1):(tmax-2)].tolist(),length),con) for m in range(resampleTime)])
                if len(temp_te_2[temp_te_2 > temp_te])/100.00 < 0.01:
                    n_teList.append(i)
                    con[(i,2)] = sample[i,(lagmax-1):(tmax-2)]
            else:
                temp_te = cmidd(x,sample[i,(lagmax+1):(tmax)],con)
                temp_te_2 = np.array([cmidd(x,random.sample(sample[i,(lagmax+1):(tmax)].tolist(),length),con) for m in range(resampleTime)])
                if len(temp_te_2[temp_te_2 > temp_te])/100.00 < 0.01:
                    n_teList.append(i)
                    con[(i,1)] = sample[i,(lagmax+1):(tmax)]

        lag_max_pre[n,n_teList] += 1

        ####remove lag&pc####
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
    np.savetxt(conf.get_filename_via_tpl(
        'cse_lag_max_pre', n_users=nnode, n_samples=tmax), lag_max_pre, fmt='%d', delimiter=',')
    np.savetxt(conf.get_filename_via_tpl(
        'cse_lag_max_late', n_users=nnode, n_samples=tmax), lag_max_late, fmt='%d', delimiter=',')
    np.savetxt(conf.get_filename_via_tpl(
        'cse_lag_te', n_users=nnode, n_samples=tmax), lag_te, delimiter=',')
    logger.info('CSE results have been saved in result folder.')
    return lag_max_pre,lag_max_late,lag_te

def testTE(lagmax,tmax,resampleTime,nnode,sample,indegreeaverage):
    """
    function of TE
    """
    length = tmax - lagmax-1
    lag_max_pre = np.eye(nnode, dtype=int)
    lag_max_late = np.zeros((nnode,nnode),dtype=np.int)
    lag_te = np.zeros((nnode,nnode))
    # sample = sample.T
    for n in range(nnode):
        con = {}
        con[(n,1)] = sample[n,lagmax:tmax-1]
        x = sample[n,lagmax+1:tmax]
        nodeset = range(nnode)
        ####find lag&pc####
        for i in nodeset:
            if n==i:
                temp_te = cmidd(x,sample[i,(lagmax-1):(tmax-2)],con)
                temp_te_2 = np.array([cmidd(x,random.sample(sample[i,(lagmax-1):(tmax-2)].tolist(),length),con) for m in range(resampleTime)])
                if len(temp_te_2[temp_te_2 > temp_te])/100.00 < 0.01:
                    lag_max_late[i,n] += 1
                    lag_max_pre[i,n] += 1
                    lag_te[n,i] = temp_te
            else:
                temp_te = cmidd(x,sample[i,(lagmax+1):(tmax)],con)
                temp_te_2 = np.array([cmidd(x,random.sample(sample[i,(lagmax+1):(tmax)].tolist(),length),con) for m in range(resampleTime)])
                if len(temp_te_2[temp_te_2 > temp_te])/100.00 < 0.01:
                    lag_max_late[i,n] += 1
                    lag_max_pre[i,n] += 1
                    lag_te[n,i] = temp_te

    for n in range(nnode):
        for i in range(nnode):
            if n != i :
                if (lag_te[n,i]>=lag_te[i,n]):
                    lag_max_late[i,n] = 0

                elif lag_te[i,n]:
                    lag_max_late[n,i] = 0
            else:
                break
    np.savetxt(conf.get_filename_via_tpl(
        'te_lag_max_pre', n_users=nnode, n_samples=tmax), lag_max_pre, fmt='%d', delimiter=',')
    np.savetxt(conf.get_filename_via_tpl(
        'te_lag_max_late', n_users=nnode, n_samples=tmax), lag_max_late, fmt='%d', delimiter=',')
    np.savetxt(conf.get_filename_via_tpl(
        'te_lag_te', n_users=nnode, n_samples=tmax), lag_te, delimiter=',')
    logger.info('TE results have been saved in result folder.')
    return lag_max_pre,lag_max_late,lag_te
