# coding: utf-8
"""
Created on Jan 31 Wed 2017

@author: Yong Xie
"""
import os

import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
import pandas as pd

from sklearn.metrics import roc_curve, auc

from lib.entropy_estimators import cmi
import utils.config_util as conf
from utils.config_util import get_console_logger

logger = get_console_logger(__name__)

N_USERS = conf.N_USERS
N_SAMPLES = conf.N_SAMPLES
N_DIMS = conf.N_DIMS


def te_transform(data, lag=1):
    """
    Perform transfer entropy on each pair of samples to find out causal relationships.
    :param data: 3d array. Each element is a matrix of samples of a user.
    :param lag: Length of lag. Default is 1.
    :return: Causal network and te matrix.
    """
    data = np.array(data)
    n_nodes = data.shape[0]
    cn = np.zeros((n_nodes, n_nodes))
    te_mat = np.zeros((n_nodes, n_nodes))

    # Calculate te and fill with the causal network.
    for i in range(n_nodes):
        sample_i = data[i]
        for j in range(i, n_nodes):
            sample_j = data[j]
            # Construct variables XP, YP and X/YF for te estimator.
            sample_i_p = sample_i[lag:]
            sample_i_f = sample_i[:-lag]
            sample_j_p = sample_j[lag:]
            te_i_j = cmi(sample_i_f, sample_j_p, sample_i_p)
            te_mat[i][j] = te_i_j
            if i != j:
                sample_j_f = sample_j[:-lag]
                te_j_i = cmi(sample_j_f, sample_i_p, sample_j_p)
                te_mat[j][i] = te_j_i
    return cn, te_mat


def evaluate(n_users, n_samples, n_dims):
    """
    Evaluate result via precise, recall and f-value.
    :return: Accuracy, recall and f1.
    """
    result = np.loadtxt(conf.get_data_filename_via_template('te', n_users=n_users, n_samples=n_samples, n_dims=n_dims),
                        delimiter=',')
    # print(result)
    new_result = np.zeros(result.shape)
    new_result_state = np.zeros(result.shape).astype(int)
    for i in range(n_users):
        for j in range(n_users):
            if result[i][j] > 0.1:
                new_result[i][j] = result[i][j]
                new_result_state[i][j] = 1
    # for i in range(n_users):
    #     for j in range(i, n_users):
    #         if abs(result[i][j]-result[j][i]) < 0.1:
    #             new_result[i][j] = new_result[j][i] = 0
    # print(new_result)
    with open(conf.get_data_filename_via_template('re', n_users=n_users, n_samples=n_samples, n_dims=n_dims), 'w') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerows(new_result)

    comparison = np.loadtxt(conf.RESULT_DIR + '/transfer', delimiter=',', dtype=int)
    # print(comparison)

    print('----Evaluation of %d users, %d samples, %d dims. ----' % (n_users, n_samples, n_dims))
    acc_rate = np.sum(new_result_state == comparison) * 1. / np.power(n_users, 2)
    predict_result = np.zeros((2, 2)).astype(int)

    for i in range(n_users):
        for j in range(n_users):
            predict_result[~comparison[i][j]][~new_result_state[i][j]] += 1
    print(predict_result)

    p = 1. * predict_result[0][0] / (np.sum(predict_result[:, 0]))
    r = 1. * predict_result[0][0] / (np.sum(predict_result[0, :]))
    f1 = (2*p*r) / (p+r)
    print('Accuracy: %.3f' % acc_rate)
    print('Precise: %.3f' % p)
    print('Recall: %.3f' % r)
    print('F1: %.3f' % f1)

    print('---ROC-AUC---')
    fpr, tpr, thresholds = roc_curve(comparison.reshape(n_users * n_users), new_result.reshape(n_users * n_users))
    print('fpr: ')
    print(fpr)
    print('tpr: ')
    print(tpr)
    print('thresholds: ')
    print(thresholds)

    roc_auc = auc(fpr, tpr)
    print('roc-auc:')
    print(roc_auc)

    print('\n\n')

if __name__ == '__main__':
    # task: 0 -- infer; 1 -- evaluate;
    # task = 0
    task = 1

    if task == 1:
        dims = [500, 800, 1000]
        for n_dims in dims:
            evaluate(N_USERS, N_SAMPLES, n_dims)
        exit(0)

    data = []
    idx = 0
    with open(conf.get_data_filename_via_template('lsi', n_users=N_USERS, n_samples=N_SAMPLES, n_dims=N_DIMS)) as fp:
        csv_reader = csv.reader(fp)
        samples = np.zeros((N_SAMPLES, N_DIMS))
        for line in csv_reader:
            if len(line):
                if len(line) < N_DIMS:
                    line = np.array(line.extend([0.0]*(N_DIMS - len(line))))
                samples[idx] = line

            idx += 1
            if idx == N_SAMPLES:
                data.append(samples)
                samples = np.zeros((N_SAMPLES, N_DIMS))
                idx = 0

    data = np.array(data)
    print(data.shape)

    causal_network, te_matrix = te_transform(data)
    with open(conf.get_data_filename_via_template('te', n_users=N_USERS, n_samples=N_SAMPLES, n_dims=N_DIMS), 'w') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerows(te_matrix)
        logger.info(conf.get_data_filename_via_template('te', n_users=N_USERS, n_samples=N_SAMPLES, n_dims=N_DIMS) + ' saved.')




