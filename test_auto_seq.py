# coding: utf-8
import numpy as np

import utils.config_util as conf
import utils.entropy_estimators as ee
from gen_data import DataGenerator, MblogInfo

# Read sequences of all users which have been processed with the smallest time step.
data_info = {'n_users': 12, 'n_samples': 2192, 'n_dims': 100}
seq_filename = conf.get_filename_via_tpl('seq', n_users=data_info['n_users'], n_samples=data_info['n_samples'])
sequences = np.loadtxt(seq_filename, np.int, delimiter=',')
# print(sequences)
# print(sequences.shape)

# Set an active rate and find the optimal time steps to make a maximum joint entropy.
active_rate = .4
active_count = active_rate * data_info['n_users']

hist = sequences.sum(0)
active_status = np.zeros(sequences.shape[1])
active_status[np.where(hist > active_count)] = 1

# print(active_status)
# print(len(active_status))
# print(sum(active_status))

last_joint_entropy = None


def merge_sub_sequence(s, e, seqs):
    global last_joint_entropy

    if not last_joint_entropy:
        # Calculate joint entropy before merge.
        last_joint_entropy = ee.entropyd(seqs.T.tolist())

    # Calculate joint entropy after merge.
    for row in seqs:
        if sum(row[s:e]) > 0:
            row[e-1] = 1
        else:
            row[e-1] = 0
    new_joint_entropy = ee.entropyd(np.concatenate((seqs[:, :start], seqs[:, end - 1:]), axis=1).T.tolist())

    # Return whether merge or not.
    if new_joint_entropy < last_joint_entropy:
        last_joint_entropy = new_joint_entropy
        # print(new_joint_entropy)
        return True
    return False

time_steps = []
start = end = 0

while True:
    if start >= sequences.shape[1]:
        break

    while end < sequences.shape[1] and active_status[end] == 0:
        end += 1
    time_step = 1 if start == end else end - start
    if time_step == 1:
        time_steps.append(time_step)
        if start == end:
            end += 1
    elif merge_sub_sequence(start, end, sequences):
        time_steps.append(time_step)
        sequences = np.concatenate((sequences[:, :start], sequences[:, end - 1:]), axis=1)
        end -= (time_step - 1)
        print('merge.', last_joint_entropy)
    else:
        print('Not merge.', last_joint_entropy)
        time_steps.append(1)
        end = start + 1
    start = end

print(time_steps)
print(len(time_steps))
print(sum(time_steps))

# Auto justify the active rate to get an optimal one and the corresponding time steps.


# Construct new sequences and text list.
mblog_info = MblogInfo('mblog', '10.21.50.32', 27017, 'user_social', 'Mblog', [],
                           '2011-10-01', '2017-10-01', '%Y-%m-%d', [i * 24*3600 for i in time_steps])

# DG = DataGenerator(mblog_info)
# DG.construct_time_series_data()

