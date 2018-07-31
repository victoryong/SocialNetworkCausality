# coding: utf-8
import numpy as np
import csv

import utils.config_util as conf
from utils import model_object as mo
from utils.log import get_console_logger

np.set_printoptions(threshold=np.inf)
logger = get_console_logger(__name__)

n_users, n_samples = 12, 2192
original_seq = np.loadtxt(conf.get_filename_via_tpl('seq', n_users=12, n_samples=2192), delimiter=',')
print(original_seq.shape)


def segment_ts_enum():
    """
    Segment time series with time steps constructed by inserting numbers one by one the the time steps list.
    """
    times_steps = {original_seq.shape[1]}
    results = [['obj_func', 'h_seq', 'it', 'penalty']]

    while len(times_steps) < original_seq.shape[1]:
        max_obj = max_idx = -1
        max_line = None

        for i in range(1, original_seq.shape[1]):
            if i not in times_steps:
                temp_steps = [i] + list(times_steps)
                temp_steps.sort()

                obj, h_seq, it, te_mat, penalty = mo.cal_object_function(original_seq, temp_steps, lamb=.05, return_details=True)

                if obj > max_obj:
                    max_obj = obj
                    max_idx = i
                    # max_line = [temp_steps, obj, h_seq, te_mat, complexity]
                    max_line = [obj, h_seq, it, penalty]
                    max_line = [str(i) for i in max_line]

        results.append(max_line)
        times_steps.add(max_idx)
        print(max_obj, max_idx)

    with open('search_timesteps_results_no_permutationtest.csv', 'w', newline='') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerows(results)


def segment_ts_inner_pro(thres_low=1, thres_high=n_users+1):
    """
    Segment time series using a bottom-up method, and its costs are calculated as inner products of every two neighbor time points.
    """
    seq = original_seq.copy()
    costs = np.zeros(seq.shape[1] - 1)
    time_steps = np.array(range(1, seq.shape[1] + 1))
    results = [['time_steps', 'obj_func', 'h_seq', 'it', 'te_mat', 'penalty', 'len']]

    for i in range(seq.shape[1] - 1):
        costs[i] = np.dot(seq[:, i], seq[:, i + 1])

    # n_users = 2
    for nu in range(thres_low, thres_high):
        cost_thres = nu
        # cost_thres = 6

        min_cost = min(costs)
        length = costs.shape[0]
        while min_cost < cost_thres:
            # print(min_cost)
            i = 0
            while i < length:
                if costs[i] == min_cost:
                    # Merge sequences of i and i+1
                    seq[:, i] = seq[:, i] + seq[:, i+1]
                    seq[seq[:, i] > 0, i] = 1
                    seq = np.delete(seq, i+1, axis=1)
                    length -= 1

                    # recalculate costs[i-1] and costs[i]
                    if i > 0:
                        costs[i-1] = np.dot(seq[:, i-1], seq[:, i])
                    if i < length:
                        costs[i] = np.dot(seq[:, i], seq[:, i+1])
                        costs = np.delete(costs, i+1)
                    elif i == length:
                        costs = np.delete(costs, i)

                    # update time steps
                    time_steps[i] = time_steps[i+1]
                    time_steps = np.delete(time_steps, i+1)
                i += 1
            min_cost = min(costs)

        # print(costs)
        print(len(costs))

        # print(time_steps)
        # print(len(time_steps))

        obj, h_seq, it, te_mat, penalty = mo.cal_object_function(original_seq, time_steps, lamb=.005, return_details=True)
        results.append([time_steps, obj, h_seq, it, te_mat, penalty, len(time_steps)])

    with open('search_timesteps_results_inner_pro.csv', 'w', newline='') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerows(results)

    return results


if __name__ == '__main__':
    segment_ts_inner_pro()

