# coding: utf-8
import numpy as np

import utils.entropy_estimators as ee
from utils.log import get_console_logger

logger = get_console_logger(__name__)


def make_seq_via_time_steps(seq, time_steps):
    """
    Construct new sequences according to time steps.
    :param seq: Original sequences that is to be reconstructed.
    :param time_steps: A list of time steps.
    :return: New sequences.
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


def cal_object_function(seq, time_steps, lamb=0, return_details=False):
    """
    Make new sequences and calculate object function which we would like to maximise.
    :param seq: Sequences with smallest time steps.
    :param time_steps: A list of time steps.
    :param lamb: Super parameter, the weight of model complexity.
    :param return_details: Whether to return details or just simple result.
    :return: Object function result.
    """
    new_seq = make_seq_via_time_steps(seq, time_steps)

    # Joint Entropy of new sequences
    h_seq = ee.entropyd(new_seq.T.tolist())
    # Information Transfer of each pairs of variables.
    te_mat = ee.transfer_entropyd(new_seq)
    it = te_mat.sum()
    # Model complexity
    l = len(time_steps)
    # complexity = l if l < seq.shape[1] / 2 else seq.shape[1] - l
    complexity = l
    return h_seq + it + lamb * complexity, h_seq, it, te_mat, complexity if return_details else h_seq + it + lamb * complexity



