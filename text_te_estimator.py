# coding: utf-8
"""
Created on Jan 31 Wed 2017

@author: Yong Xie
"""

import numpy as np
import csv

from sklearn.metrics import roc_curve, auc

from utils.entropy_estimators import cmi, cond_entropy, entropy
import utils.config_util as conf
from utils.config_util import get_console_logger
from optimize_obj_func import get_n_samples_list
from text_processing import TextProcessor

logger = get_console_logger(__name__)

N_USERS = conf.N_USERS
N_SAMPLES = conf.N_SAMPLES
N_DIMS = conf.N_DIMS


def calculate_te(data, vec_type, lag=1, normalised=True):
    """
    Perform transfer entropy on each pair of samples to find out causal relationships.
    """
    data = np.array(data)
    n_nodes, n_samples, n_dims = data.shape
    cn = np.zeros((n_nodes, n_nodes))
    te_mat = np.zeros((n_nodes, n_nodes))
    #
    if normalised:
        H_0 = np.zeros(n_nodes)
        for i in range(n_nodes):
            max_min = np.max(data[i], 0) - np.min(data[i], 0)
            H_0[i] = np.sum(np.log2(max_min))
        # for i in range(n_nodes):
        #     H_0[i] = entropy(data[i])

    logger.info('Calculating te...')
    # Calculate te and fill with the causal network.
    for i in range(n_nodes):
        sample_i = data[i]
        for j in range(i, n_nodes):
            sample_j = data[j]
            # Construct variables XP, YP and X/YF for te estimator.
            sample_i_p = sample_i[lag:]
            sample_i_f = sample_i[:-lag]
            sample_j_p = sample_j[lag:]
            te_j_i = cmi(sample_i_f, sample_j_p, sample_i_p)

            if normalised:
                te_j_i = te_j_i / (H_0[i] - cond_entropy(sample_i_f, np.concatenate((sample_i_p, sample_j_p), 1)))
                # te_j_i = te_j_i / H_0[i]
            te_mat[j][i] = te_j_i

            if i != j:
                sample_j_f = sample_j[:-lag]
                te_i_j = cmi(sample_j_f, sample_i_p, sample_j_p)
                if normalised:
                    te_i_j = te_i_j / (H_0[j] - cond_entropy(sample_j_f, np.concatenate((sample_i_p, sample_j_p), 1)))
                    # te_i_j = te_i_j / H_0[j]
                te_mat[i][j] = te_i_j
    te_path = conf.get_filename_via_tpl(
        'te_' + vec_type, n_users=n_nodes, n_samples=n_samples, n_dims=n_dims, lag=lag)
    np.savetxt(te_path, te_mat, delimiter=',', fmt='%f')
    logger.info('Te result has been saved in %s. ' % te_path)
    return cn, te_mat


def evaluate(n_users, n_samples, n_dims):
    """
    Evaluate result via precise, recall and f-value.
    :return: Accuracy, recall and f1.
    """
    result = np.loadtxt(conf.get_filename_via_tpl('te_text', n_users=n_users, n_samples=n_samples, n_dims=n_dims), delimiter=',')
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
    with open(conf.get_filename_via_tpl('re', n_users=n_users, n_samples=n_samples, n_dims=n_dims), 'w') as fp:
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


def calculate_text_te(n_samples_list=None, normalised=True):
    n_users, n_dims = 12, 50
    if not n_samples_list:
        n_samples_list = get_n_samples_list()

    # n_samples_list = n_samples_list[2:8]
    print(n_samples_list)
    for n_samples in n_samples_list:
        data_info = {'n_users': n_users, 'n_samples': n_samples, 'n_dims': n_dims}

        # use the dict of data info defined above.
        tp = TextProcessor(data_info['n_users'], data_info['n_samples'], data_info['n_dims'])

        data = tp.load_vec('lda')
        print(data.shape)
        cn, te_mat = calculate_te(data, 'lda', normalised=normalised)
        print('te_lda_%d_%d_%d' % (data_info['n_users'], data_info['n_samples'], data_info['n_dims']))
        print(cn)
        print(te_mat)


if __name__ == '__main__':
    # # task: 0 -- infer; 1 -- evaluate;
    # # task = 0
    # task = 1
    #
    # if task == 1:
    #     dims = [100]
    #     for n_dims in dims:
    #         evaluate(N_USERS, N_SAMPLES, n_dims)
    #     exit(0)
    #
    # data = []
    # idx = 0
    # with open(conf.get_filename_via_tpl('w2v', n_users=N_USERS, n_samples=N_SAMPLES, n_dims=N_DIMS)) as fp:
    #     csv_reader = csv.reader(fp)
    #     samples = np.zeros((N_SAMPLES, N_DIMS))
    #     for line in csv_reader:
    #         if len(line):
    #             if len(line) < N_DIMS:
    #                 line = np.array(line.extend([0.0]*(N_DIMS - len(line))))
    #             samples[idx] = line
    #
    #         idx += 1
    #         if idx == N_SAMPLES:
    #             data.append(samples)
    #             samples = np.zeros((N_SAMPLES, N_DIMS))
    #             idx = 0
    #
    # data = np.array(data)
    # print(data.shape)
    #
    # causal_network, te_matrix = calculate_te(data)
    # with open(conf.get_filename_via_tpl('te', n_users=N_USERS, n_samples=N_SAMPLES, n_dims=N_DIMS), 'w') as fp:
    #     csv_writer = csv.writer(fp)
    #     csv_writer.writerows(te_matrix)
    #     logger.info(conf.get_filename_via_tpl('te', n_users=N_USERS, n_samples=N_SAMPLES, n_dims=N_DIMS) + ' saved.')

    calculate_text_te([2192, 2166, 1411])

