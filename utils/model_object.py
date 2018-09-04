# coding: utf-8
from functools import reduce
import numpy as np

import utils.entropy_estimators as ee
from utils.log import get_console_logger

logger = get_console_logger(__name__)
np.set_printoptions(threshold=np.inf)


def make_seq_via_time_steps(seq, time_steps):
    """
    Construct new sequences according to time steps.
    """
    # nidx = oidx = 0
    # for step in time_steps:
    #     if step == 1:
    #         if nidx != oidx:
    #             seq[:, nidx] = seq[: oidx]
    #     else:
    #         for row in seq:
    #             row[nidx] = sum(row[oidx: oidx+step])
    #     oidx += step
    #     nidx += 1
    # if nidx != len(time_steps):
    #     logger.error('nidx != len(time_steps)')
    # if oidx != seq.shape[1]:
    #     raise IndexError('Total amount of time steps is smaller than sample length.')
    # return seq[:, :len(time_steps)]
    new_seq = np.zeros((seq.shape[0], len(time_steps)))
    nidx = oidx = 0
    for time_point in time_steps:
        step = time_point - oidx
        if step == 1:
            new_seq[:, nidx] = seq[:, oidx]
        else:
            for r in range(seq.shape[0]):
                new_seq[r, nidx] = sum(seq[r, oidx: oidx+step])
            new_seq[new_seq[:, nidx] > 0, nidx] = 1
        oidx += step
        nidx += 1
    if nidx != len(time_steps):
        logger.error('nidx != len(time_steps)')
    if oidx != seq.shape[1]:
        raise IndexError('Total amount of time steps is smaller than sample length.')
    return new_seq

pre_computed = False
one_counts = None  # One's counts of each variable in current level
h_vars_nl = None  # Two possible results of each variable in next level
h_tps_nl = None  # Each time point's entropy in current level
tri_val_counts = None  # Possible value Counts' of every triple variables for transfer entropy calculation


# Update tri_val_counts without recalculation
def reduce_tri_val_counts_elem(l_seq, i, j, idx, pos, counts):
    assert len(pos) == 3 and '11' in pos, "pos error"
    if pos[0] == '1':
        counts[i][j][int(l_seq[j][idx - 1] * 4 + l_seq[i][idx] * 2 + l_seq[j][idx])] -= 1
        counts[j][i][int(l_seq[i][idx - 1] * 4 + l_seq[j][idx] * 2 + l_seq[i][idx])] -= 1
    if pos[1] == '1':
        counts[i][j][int(l_seq[j][idx] * 4 + l_seq[i][idx + 1] * 2 + l_seq[j][idx + 1])] -= 1
        counts[j][i][int(l_seq[i][idx] * 4 + l_seq[j][idx + 1] * 2 + l_seq[i][idx + 1])] -= 1
    if pos[2] == '1':
        counts[i][j][int(l_seq[j][idx + 1] * 4 + l_seq[i][idx + 2] * 2 + l_seq[j][idx + 2])] -= 1
        counts[j][i][int(l_seq[i][idx + 1] * 4 + l_seq[j][idx + 2] * 2 + l_seq[i][idx + 2])] -= 1


def add_tri_val_counts_elem(l_seq, merged_col, i, j, idx, pos, counts):
    assert len(pos) == 2 and '1' in pos, 'pos error'
    if pos[0] == '1':
        counts[i][j][int(l_seq[j][idx - 1] * 4 + merged_col[i] * 2 + merged_col[j])] += 1
        counts[j][i][int(l_seq[i][idx - 1] * 4 + merged_col[j] * 2 + merged_col[i])] += 1
    if pos[1] == '1':
        counts[i][j][int(merged_col[j] * 4 + l_seq[i][idx + 2] * 2 + l_seq[j][idx + 2])] += 1
        counts[j][i][int(merged_col[i] * 4 + l_seq[j][idx + 2] * 2 + l_seq[i][idx + 2])] += 1


def pre_compute(seq, time_steps, delete_idx=-1, last_seq=None):
    """
    Compute some entropies for next level to fasten object function calculation.
    """
    global pre_computed, h_vars_nl, one_counts, h_tps_nl, tri_val_counts
    n_vars, n_tps = seq.shape[0], time_steps.shape[0]

    if delete_idx == -1:
        new_seq = seq if len(time_steps) == seq.shape[1] else make_seq_via_time_steps(seq, time_steps)

        # Each variable may has two possible states in next level, losing either a '0' or a '1'
        h_vars_nl = np.zeros((n_vars, 2))
        one_counts = np.zeros(n_vars)
        for i in range(n_vars):
            one_counts[i] = np.sum(new_seq[i])
            if 0 < one_counts[i] < n_tps:
                h_vars_nl[i, 0] = ee.entropyfromprobs(map(lambda z: float(z) / (n_tps - 1), [one_counts[i], n_tps - 1 - one_counts[i]]), base=2)
                h_vars_nl[i, 1] = ee.entropyfromprobs(map(lambda z: float(z) / (n_tps - 1), [one_counts[i] - 1, n_tps - one_counts[i]]), base=2)

        # Each time point has an entropy, and then the sum of all is saved in h_tps_nl[n_tps]
        h_tps_nl = np.zeros(n_tps + 1)
        for i in range(n_tps):
            h_tps_nl[i] = ee.entropyd(new_seq[:, i].tolist())
            h_tps_nl[-1] += h_tps_nl[i]

        # Each transfer entropy requires a triple variables and this triple may have three possible values
        tri_val_counts = np.zeros((n_vars, n_vars, 8), np.int)
        for i in range(n_vars - 1):
            for j in range(i + 1, n_vars):
                for k in range(n_tps - 1):
                    tri_val_counts[i][j][int(new_seq[j][k]*4 + new_seq[i][k+1]*2 + new_seq[j][k+1])] += 1
                    tri_val_counts[j][i][int(new_seq[i][k]*4 + new_seq[j][k+1]*2 + new_seq[i][k+1])] += 1

    else:
        seq_i = last_seq[:, delete_idx].reshape(n_vars).astype(int)
        seq_i1 = last_seq[:, delete_idx+1].reshape(n_vars).astype(int)
        delete_ele = seq_i & seq_i1
        merged_col = seq_i | seq_i1

        # Update one_counts and h_vars_nl
        one_counts = one_counts - delete_ele

        for i in range(n_vars):
            if 0 < one_counts[i] < n_tps:
                h_vars_nl[i, 0] = ee.entropyfromprobs(
                    map(lambda z: float(z) / (n_tps - 1), [one_counts[i], n_tps - 1 - one_counts[i]]), base=2)
                h_vars_nl[i, 1] = ee.entropyfromprobs(
                    map(lambda z: float(z) / (n_tps - 1), [one_counts[i] - 1, n_tps - one_counts[i]]), base=2)

        # Update h_tps_nl without recalculation of all time points
        h_tps_nl[-1] -= (h_tps_nl[delete_idx] + h_tps_nl[delete_idx + 1])
        h_tps_nl = np.delete(h_tps_nl, delete_idx)
        h_tps_nl[delete_idx] = ee.entropyd(merged_col.tolist())
        h_tps_nl[-1] += h_tps_nl[delete_idx]

        # Update tri_val_counts
        if delete_idx == 0:
            for i in range(n_vars - 1):
                for j in range(i+1, n_vars):
                    reduce_tri_val_counts_elem(last_seq, i, j, delete_idx, '011', tri_val_counts)
                    add_tri_val_counts_elem(last_seq, merged_col, i, j, delete_idx, '01', tri_val_counts)
                    
        elif delete_idx == n_tps - 1:
            for i in range(n_vars - 1):
                for j in range(i+1, n_vars):
                    reduce_tri_val_counts_elem(last_seq, i, j, delete_idx, '110', tri_val_counts)
                    add_tri_val_counts_elem(last_seq, merged_col, i, j, delete_idx, '10', tri_val_counts)
        else:
            for i in range(n_vars - 1):
                for j in range(i+1, n_vars):
                    reduce_tri_val_counts_elem(last_seq, i, j, delete_idx, '111', tri_val_counts)
                    add_tri_val_counts_elem(last_seq, merged_col, i, j, delete_idx, '11', tri_val_counts)

    # if not pre_computed:
    #     print(tri_val_counts)
    pre_computed = True


def object_function(seq, time_steps, lamb=0, return_details=False, delete_idx=-1, last_seq=None):
    """
    Make new sequences and calculate object function which we would like to maximise.
    """
    n_vars, n_tps = seq.shape[0], time_steps.shape[0]

    new_seq = seq if n_tps == seq.shape[1] else make_seq_via_time_steps(seq, time_steps)

    # Some values have been pre-computed
    if pre_computed and h_tps_nl is not None and h_tps_nl.shape[0] == n_tps + 2:
        # logger.info('Pre-computed!!!!')
        # print(h_vars_nl)
        # print(h_tps_nl)
        # print(one_counts)

        seq_i = last_seq[:, delete_idx].reshape(n_vars).astype(int)
        seq_i1 = last_seq[:, delete_idx + 1].reshape(n_vars).astype(int)
        delete_ele = seq_i & seq_i1
        merged_col = seq_i | seq_i1

        # Summation entropy of each variable
        h_vars = 0
        for i in range(n_vars):
            h_vars += h_vars_nl[i, delete_ele[i]]

        # Summation entropy of each time point
        h_tps = h_tps_nl[-1] - h_tps_nl[delete_idx] - h_tps_nl[delete_idx + 1] + ee.entropyd(merged_col)

        # Information Transfer of each pairs of variables.
        te_mat = np.zeros((n_vars, n_vars))
        
        temp_tri_counts = tri_val_counts.copy()
        if delete_idx == 0:
            for i in range(n_vars - 1):
                for j in range(i + 1, n_vars):
                    reduce_tri_val_counts_elem(last_seq, i, j, delete_idx, '011', temp_tri_counts)
                    add_tri_val_counts_elem(last_seq, merged_col, i, j, delete_idx, '01', temp_tri_counts)

        elif delete_idx == n_tps - 1:
            for i in range(n_vars - 1):
                for j in range(i + 1, n_vars):
                    reduce_tri_val_counts_elem(last_seq, i, j, delete_idx, '110', temp_tri_counts)
                    add_tri_val_counts_elem(last_seq, merged_col, i, j, delete_idx, '10', temp_tri_counts)
        else:
            for i in range(n_vars - 1):
                for j in range(i + 1, n_vars):
                    reduce_tri_val_counts_elem(last_seq, i, j, delete_idx, '111', temp_tri_counts)
                    add_tri_val_counts_elem(last_seq, merged_col, i, j, delete_idx, '11', temp_tri_counts)

        def cal_te_elem(i, j):
            this_counts = temp_tri_counts[i][j]
            ent_xyz = ee.entropyfromprobs(map(lambda c: float(c) / (n_tps - 1), this_counts))
            ent_z = ee.entropyfromprobs(map(
                lambda c: float(c) / (n_tps - 1), [sum(this_counts[::2]), sum(this_counts[1::2])]))
            ent_xz = ee.entropyfromprobs(map(
                lambda cs: float(cs[0] + cs[1]) / (n_tps - 1), zip(this_counts[[0, 1, 4, 5]], this_counts[[2, 3, 6, 7]])))
            ent_yz = ee.entropyfromprobs(map(
                lambda cs: float(cs[0] + cs[1]) / (n_tps - 1), zip(this_counts[:4], this_counts[4:])))
            te_mat[i, j] = ent_xz + ent_yz - ent_xyz - ent_z
            # print(ent_xyz, ent_z, ent_xz, ent_yz)

        for i in range(n_vars - 1):
            for j in range(i+1, n_vars):
                cal_te_elem(i, j)
                cal_te_elem(j, i)

        # if n_tps == 6:
        #     print(temp_tri_counts)
        #     print('----')

    # Current level's pre-computation hasn't been done
    else:
        # Summation entropy of each variable
        # h_seq = ee.entropyd(new_seq.T.tolist())
        # h_vars = 0
        # for i in new_seq:
        #     h_vars += ee.entropyd(i.T.tolist())

        h_vars = sum([ee.entropyd(i.tolist()) for i in new_seq])

        # Summation entropy of each time point
        h_tps = sum([ee.entropyd(new_seq[:, i].tolist()) for i in range(new_seq.shape[1])])

        # Information Transfer of each pairs of variables.
        te_mat = ee.transfer_entropyd(new_seq)
    it = te_mat.sum()

    # Regularization
    regularization = n_tps * lamb

    rst_terms = [100 * h_vars / n_vars, 100 * h_tps / n_tps, 12.5 * it / (n_vars * (n_vars - 1)) + 50, regularization]
    return sum(rst_terms), new_seq, rst_terms if return_details else sum(rst_terms)



