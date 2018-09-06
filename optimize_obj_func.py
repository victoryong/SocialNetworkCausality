# coding: utf-8
import datetime
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

                obj, h_seq, it, te_mat, penalty = mo.object_function(original_seq, temp_steps, lamb=.05, return_details=True)

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


def segment_ts_bottom_up(pre_compute=True):
    """
    Segment time series using a bottom-up method, and its candidate_obj are calculated as inner products of every two neighbor time points.
    """
    start_time = datetime.datetime.now()

    # seq = original_seq.copy()
    seq = original_seq[:, :100].copy()
    # print(seq)
    results = [['time_steps', 'obj_func', 'h_vars', 'h_tps', 'it', 'regularization']]
    # lamb = -0.01
    lamb = -100 / seq.shape[1]

    # Initial
    last_time_steps, last_idx = np.array(range(1, seq.shape[1] + 1)), -1
    last_obj, last_seq, last_obj_details = mo.object_function(seq, last_time_steps, lamb, True)
    results.append([last_time_steps, last_obj] + last_obj_details)
    print('result: ', last_obj)
    print('details:', last_obj_details)
    # return

    if pre_compute:
        mo.pre_compute(seq, last_time_steps)

    # Maximize object function iteratively
    while True:
        max_idx, max_time_steps, max_candidate_obj, max_candidate_seq, max_candidate_obj_details = -1, None, -1, None, None
        next_length = last_time_steps.shape[0] - 1

        # Find maximum obj of next level's
        for i in range(next_length):
            new_time_steps = np.delete(last_time_steps, i)
            temp_candidate_obj, temp_candidate_seq, temp_candidate_obj_details = \
                mo.object_function(seq, new_time_steps, lamb, True, i, last_seq)
            # print('----')
            # print('time_steps:', new_time_steps)
            # print('temp result: ', temp_candidate_obj)
            # print('temp result terms:', temp_candidate_obj_details)

            if temp_candidate_obj > max_candidate_obj:
                max_idx, max_time_steps, max_candidate_obj, max_candidate_seq, max_candidate_obj_details = \
                    i, new_time_steps, temp_candidate_obj, temp_candidate_seq, temp_candidate_obj_details

        # Loop ending condition: None of next level's obj result greater than last level's
        if max_idx == -1 or max_candidate_obj < last_obj or last_seq.shape[1] < 4:
            break

        # Merge time points i and (i+1) to boost seq's obj result
        if pre_compute:
            mo.pre_compute(seq, max_time_steps, max_idx, last_seq)

        last_idx, last_time_steps, last_obj, last_seq, last_obj_details = \
            max_idx, max_time_steps, max_candidate_obj, max_candidate_seq, max_candidate_obj_details
        results.append([last_time_steps, last_obj] + last_obj_details)
        # print('time steps: ', last_time_steps)
        print('----\nlevel %d' % (seq.shape[1] - last_seq.shape[1]))
        print('result: ', last_obj)
        print('details:', last_obj_details)

        # break
    # return

    finish_time = datetime.datetime.now()
    print('%s用时：%f\n--------' % ('precompute' if pre_compute else 'not-precompute', (finish_time - start_time).total_seconds()))

    # Save results
    filename = conf.get_filename_via_tpl(
        'Obj', n_users=seq.shape[0], n_samples=seq.shape[1], date=datetime.datetime.now().strftime('%y%m%d%H%M%S'))
    with open(filename, 'w', newline='') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(['lambda: %f' % lamb])
        csv_writer.writerows(results)

    return results


def segment_ts_bottom_up_test(pre_compute=True):
    start_time = datetime.datetime.now()

    # seq = original_seq.copy()
    seq = original_seq[:, :100].copy()
    results = [['time_steps', 'obj_func', 'h_vars', 'h_tps', 'it', 'regularization']]
    # lamb = -0.01
    lamb = -100 / seq.shape[1]

    # Initial
    last_time_steps, last_idx = np.array(range(1, seq.shape[1] + 1)), -1
    last_obj, last_seq, last_obj_details = mo.object_function(seq, last_time_steps, lamb, True)
    results.append([last_time_steps, last_obj] + last_obj_details)
    print('result: ', last_obj)
    print('details:', last_obj_details)
    # return

    if pre_compute:
        mo.pre_compute(seq, last_time_steps)

    levels = []
    stop_level = 97
    levels_idx = 0

    max_rst = 0.
    while True:
        next_length = last_time_steps.shape[0] - 1

        # Find maximum obj of next level's
        for i in range(last_idx + 1, next_length):
            new_time_steps = np.delete(last_time_steps, i)
            temp_candidate_obj, temp_candidate_seq, temp_candidate_obj_details = \
                mo.object_function(seq, new_time_steps, lamb, True, i, last_seq)

            if temp_candidate_obj > max_rst:
                max_rst = temp_candidate_obj

            if next_length > stop_level:
                levels.append([i, new_time_steps, temp_candidate_obj, temp_candidate_seq, temp_candidate_obj_details, last_seq])
            results.append([new_time_steps, temp_candidate_obj] + temp_candidate_obj_details)

        results.append([''])
        if len(levels) <= levels_idx:
            break

        curr_level = levels[levels_idx]
        levels_idx += 1
        last_idx, last_time_steps, last_obj, last_seq, last_obj_details, ll_seq = \
            curr_level[0], curr_level[1], curr_level[2], curr_level[3], curr_level[4], curr_level[5]
        # Merge time points i and (i+1) to boost seq's obj result
        if pre_compute:
            mo.pre_compute(seq, last_time_steps)

        # break
    # return

    finish_time = datetime.datetime.now()
    print('%s用时：%f\n--------' % ('precompute' if pre_compute else 'not-precompute', (finish_time - start_time).total_seconds()))
    print('最大值: ', max_rst)

    # Save results
    filename = conf.get_filename_via_tpl(
        'Obj', n_users=seq.shape[0], n_samples=seq.shape[1], date=datetime.datetime.now().strftime('%y%m%d%H%M%S'))
    with open(filename, 'w', newline='') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(['lambda: %f' % lamb])
        csv_writer.writerows(results)

    return results


def get_n_samples_list():
    n_samples_list = []
    with open('search_timesteps_results_inner_pro.csv') as fp:
        csv_reader = csv.reader(fp)
        for line in csv_reader:
            try:
                n_samples = int(line[-1])
                # print(n_samples, type(n_samples))
                if n_samples not in n_samples_list:
                    n_samples_list.append(n_samples)
            except ValueError as e:
                logger.info(e)
    return n_samples_list


def get_time_steps_list():
    time_step_list = []
    with open('search_timesteps_results_inner_pro.csv') as fp:
        csv_reader = csv.reader(fp)
        for line in csv_reader:
            try:
                time_step = eval(line[0])
                print(time_step, type(time_step))
                if time_step not in time_step_list:
                    time_step_list.append(time_step)
            except NameError as e:
                logger.info(e)
    return time_step_list


if __name__ == '__main__':
    segment_ts_bottom_up()
    # segment_ts_bottom_up(pre_compute=False)
    # segment_ts_bottom_up_test()

