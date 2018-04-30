# coding: utf-8
"""
Created on Sun Oct 29 20:55:45 2017

@author: Xie Yong

Global configurations of the entire project.
"""

import sys
import os
import psutil
from datetime import datetime
import time
import gc

from utils.log import get_console_logger

logger = get_console_logger(__name__)

# Time format
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
PUB_TIME_FORMAT = '%Y-%m-%d'
START_TIME = datetime.strptime('2011-09-30 23:59:59', TIME_FORMAT)
END_TIME = datetime.strptime('2017-09-30 23:59:59', TIME_FORMAT)

# Params
N_USERS = 10
TIME_STEP = 3600 * 24
N_SAMPLES = 2192
N_DIMS = 1000

# Data directories
ROOT = sys.path[1] if os.name == 'posix' else sys.path[0]
PRJ_NAME = 'SocialNetworkCausality'
PRJ_ROOT = str(ROOT).split(PRJ_NAME)[0] + PRJ_NAME + '/'

LIB_DIR = PRJ_ROOT + 'lib/'
DATA_DIR = PRJ_ROOT + 'data/'
RESULT_DIR = PRJ_ROOT + 'result/'

MODEL_DIR = DATA_DIR + 'model/'

# File name
FILENAME_TPL = '{file_type}{n_users}{user_id}{n_samples}{n_dims}{date}{postfix}'
MODEL_PATH_TPL = '{model_type}{n_users}{n_samples}{n_dims}{model_filename}'


def get_absolute_path(dir_type="root"):
    dir_type = dir_type.lower() if isinstance(dir_type, str) else dir_type
    return {
        'data': DATA_DIR,
        'model': MODEL_DIR,
        'lib': LIB_DIR,
        'result': RESULT_DIR
    }.get(dir_type, PRJ_ROOT)


def get_data_filename_via_template(file_type, **kwargs):
    if not isinstance(file_type, str):
        logger.error('A string for "file_type" is required. Got a(n) %s object instead. ' % type(file_type))
    file_type = file_type.lower()

    def get_params(*args):
        vals = []
        for name in args:
            try:
                if name == 'model_type':
                    vals.append(str(kwargs[name]))
                elif name == 'postfix':
                    vals.append('.' + str(kwargs[name]))
                elif name == 'model_filename':
                    vals.append('/' + str(kwargs[name]))
                else:
                    vals.append('_' + str(kwargs[name]))
            except KeyError:
                if name == 'postfix':
                    vals.append('.csv')
                elif name == 'model_filename':
                    vals.append('/' + str(vals[5]))
                else:
                    vals.append('')
        return tuple(vals)

    n_users, user_id, n_samples, n_dims, date, model_type, postfix, model_filename = get_params(
        'n_users', 'user_id', 'n_samples', 'n_dims', 'date', 'model_type', 'postfix', 'model_filename')

    filename = FILENAME_TPL.format(file_type=file_type, n_users=n_users, user_id=user_id, n_samples=n_samples,
                                   n_dims=n_dims, date=date, postfix=postfix).capitalize()
    model_path = MODEL_PATH_TPL.format(
        model_type=model_type, n_users=n_users, n_samples=n_samples, n_dims=n_dims, model_filename=model_filename)

    if file_type in ['seq', 'uid', 'text']:
        filename = DATA_DIR + filename
    elif file_type in ['lsi', 'tfidf', 'lda', 'w2v']:
        filename = DATA_DIR + file_type + '/' + filename
    elif file_type == 'model':
        filename = get_absolute_path('model') + model_path
    else:
        filename = RESULT_DIR + filename
    return filename


# Memory control
def get_memory_state():
    phymem = psutil.virtual_memory()
    line = "Memory: %5s%% %6s/%s" % (
        phymem.percent,
        str(int(phymem.used / 1024 / 1024)) + "M",
        str(int(phymem.total / 1024 / 1024)) + "M"
    )
    return line


def memory_state(time_long, percent):
    for i in range(time_long):
        mem = get_memory_state()
        if float(mem.split(' ')[2][:-1]) > percent:
            print(mem)
        time.sleep(1)


def mk_dir(path, path_type='file'):
    """
    Make dir of the directory path or the sup-directory of file path given.
    :param path: Directory or file path.
    :param path_type: 'file' or 'dir'.
    """
    if path_type == 'dir' and not os.path.exists(path):
        os.mkdir(path)
    elif path_type == 'file':
        sup_path = '/'.join(path.split('/')[:-1])
        if not os.path.exists(sup_path):
            os.mkdir(sup_path)


def delete_var(v):
    del v
    gc.collect()


if __name__ == "__main__":
    # print(get_data_filename_via_template('tfidf', n_user=12, n_dims=10, n_samples=2589))
    print(get_data_filename_via_template(
            'tfidf',
            model_type='lsi',
            n_users=N_USERS,
            n_samples=N_SAMPLES,
            n_dims=N_DIMS, postfix='mm'))
