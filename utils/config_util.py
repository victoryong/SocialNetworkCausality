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
MODEL_DIR = DATA_DIR + 'model/'
RESULT_DIR = PRJ_ROOT + 'result/'

# File name
FILENAME_TPL = '{file_type}{n_users}{user_id}{n_samples}{n_dims}.{postfix}'


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
                vals.append('_' + str(kwargs[name]))
            except KeyError:
                vals.append('')
        return tuple(vals)

    n_users, user_id, n_samples, n_dims = get_params('n_users', 'user_id', 'n_samples', 'n_dims')
    postfix = kwargs.get('postfix', 'csv')

    filename = FILENAME_TPL.format(file_type=file_type, n_users=n_users, user_id=user_id, n_samples=n_samples,
                                   n_dims=n_dims, postfix=postfix).capitalize()

    if file_type in ['seq', 'uid', 'text', 'lsi', 'tfidf']:
        filename = DATA_DIR + filename
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


if __name__ == "__main__":
    # print(get_data_filename_via_template('tfidf', n_user=12, n_dims=10, n_samples=2589))
    print(get_data_filename_via_template('t', n_users=10, n_dims=2000, n_samples=40))
