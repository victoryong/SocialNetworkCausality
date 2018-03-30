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
    print(result)
    new_result = np.zeros(result.shape)
    for i in range(n_users):
        for j in range(n_users):
            # new_result[i][j] = result[i][j]-result[j][i]
            # if new_result[i][j] < 0.1:
            #     new_result[i][j] = 0
            if result[i][j] > 0.1:
                new_result[i][j] = result[i][j]
    for i in range(n_users):
        for j in range(i, n_users):
            if abs(result[i][j]-result[j][i]) < 0.1:
                new_result[i][j] = new_result[j][i] = 0
    with open(conf.get_data_filename_via_template('re', n_users=n_users, n_samples=n_samples, n_dims=n_dims), 'w') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerows(new_result)

    comparison = np.loadtxt(conf.PRJ_DATA_ROOT + '/transfer', delimiter=',')
    print(comparison)

    p = 0
    r = 0
    f1 = 0


def test(task_type, data):
    if len(re.findall('different_n_samples', task_type)):
        # sample_1 = data[-3]
        # sample_2 = data[-1]
        lag = 1
        step = 137
        te_diff_samples_list = np.zeros((N_USERS, N_USERS, 16))

        # for m in range(data.shape[0]):
        for m in range(data.shape[0]):
            sample_1 = data[m]
            for n in range(data.shape[0]):
                sample_2 = data[n]

                for i in range(step, data.shape[1]+1, step):
                    sample_i = sample_1[:i]
                    sample_j = sample_2[:i]

                    sample_i_p = sample_i[lag:]
                    sample_i_f = sample_i[:-lag]
                    sample_j_p = sample_j[lag:]
                    sample_j_f = sample_j[:-lag]
                    te_i_j = cmi(sample_i_f, sample_j_p, sample_i_p)
                    te_j_i = cmi(sample_j_f, sample_i_p, sample_j_p)

                    te_diff_samples_list[m][n][int(i/step)-1] = te_i_j
                    te_diff_samples_list[n][m][int(i/step)-1] = te_j_i
            print('compute ended. %d' % m)

        folder_path = conf.RESULT_DIR + '/diff_{n_users}_{n_samples}_{n_dims}'.format(
            n_users=N_USERS, n_samples=N_SAMPLES, n_dims=N_DIMS)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for m in range(te_diff_samples_list.shape[0]):
            with open(folder_path + '/diff_' + str(m) + '.csv', 'w') as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerows(te_diff_samples_list[m])


def show_te_convergence(n_users, n_samples, n_dims):
    results = []
    diff_path = conf.RESULT_DIR + '/diff_{n_users}_{n_samples}_{n_dims}'.format(
        n_users=n_users, n_samples=n_samples, n_dims=n_dims)
    pic_path = conf.RESULT_DIR + '/pltPics_{n_users}_{n_samples}_{n_dims}'.format(
        n_users=n_users, n_samples=n_samples, n_dims=n_dims)
    if os.path.exists(pic_path):
        os.mkdir(pic_path)
    for i in range(n_users):
        results.append(np.loadtxt(diff_path + '/diff_' + str(i) + '.csv', delimiter=','))
    results = np.array(results)

    show_pairs = [(1, 3), (1, 4), (0, 1), (0, 2)]
    show_pairs = []

    # n_users = 1
    for i in range(n_users):
        for j in range(i, n_users):
            show_pairs.append((i, j))

    for pair in show_pairs:
        x = range(0, 2192, 137)
        plt.figure(figsize=(10, 6))
        i, j = pair[0], pair[1]
        print(results[i][j].shape)
        print(len(x))
        print(results[i][j])
        print(results[j][i])
        plt.plot(x, results[i][j], label=str(i) + '->' + str(j), marker='*')
        plt.plot(x, results[j][i], label=str(j) + '->' + str(i), marker='+')
        plt.legend(loc='upper right')
        plt.title('%d and %d' % (i, j))
        plt.savefig(pic_path + '/%dand%d.png' % (i, j))
        plt.close()
    # plt.xticks(range(137, 2192, 137), ('200504', '200912', '201108', '201306', '201502', '201610', ''))
    plt.xlabel('Sample count')
    plt.ylabel('TE')
    # plt.title('每月XX事件数')
    # plt.show()




if __name__ == '__main__':
    # task: 0 -- infer; 1 -- evaluate; 2 -- diff n samples; 3 -- convergence
    task = 0
    # task = 1
    # task = 2
    # task = 3

    if task == 1:
        evaluate(N_USERS, N_SAMPLES, N_DIMS)
        exit(0)

    if task == 3:
        show_te_convergence(N_USERS, N_SAMPLES, N_DIMS)
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

    if task == 2:
        test('different_n_samples', data)
        exit(0)

    causal_network, te_matrix = te_transform(data)
    with open(conf.get_data_filename_via_template('te', n_users=N_USERS, n_samples=N_SAMPLES, n_dims=N_DIMS), 'w') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerows(te_matrix)
        logger.info(conf.get_data_filename_via_template('te', n_users=N_USERS, n_samples=N_SAMPLES, n_dims=N_DIMS) + ' saved.')




