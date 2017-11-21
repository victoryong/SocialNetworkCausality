# coding: utf-8
"""
Created on Sun Oct 29 20:55:45 2017

@author: Xie Yong

Global configurations of the entire project.
"""

import sys
import os
from pathlib import PurePath
import psutil
from datetime import datetime

from utils.log import get_console_logger

logger = get_console_logger(__name__)

# Time format
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
PUB_TIME_FORMAT = '%Y-%m-%d'
START_TIME = datetime.strptime('2011-10-31 23:59:59', TIME_FORMAT)
END_TIME = datetime.strptime('2017-10-31 23:59:59', TIME_FORMAT)


# Data directories
ROOT = sys.path[1] if os.name == 'posix' else sys.path[0]
_PP_ROOT = PurePath(ROOT)
PRJ_DATA_ROOT = str(_PP_ROOT.parent) + '/Data/' + str(_PP_ROOT.name)
SINA_DATA_ROOT = str(_PP_ROOT.parent) + '/Data/sina_weibo_data'
USER_DATA_DIR = PRJ_DATA_ROOT + '/user_data'

# File name
UID_FILE = 'UserId_{n_users}_{n_samples}.csv'
SEQ_FILE = 'Seq_{n_users}_{n_samples}.csv'
TEXT_FILE = 'Text_{userid}_{n_samples}.csv'
TF_IDF_FILE = 'TFIDF_{n_users}_{n_samples}.csv'


def get_root_dir(root_type="ROOT"):
    if isinstance(root_type, str):
        root_type = root_type.upper()
    elif not isinstance(root_type, int):
        raise ValueError('Input that allows is a root type(str) or flag(int)!')

    if root_type == 'DATA_ROOT' or root_type == 1:
        return PRJ_DATA_ROOT
    elif root_type == 'SINA_ROOT' or root_type == 2:
        return SINA_DATA_ROOT
    else:
        return ROOT


def get_sina_file_path(file_name="mblog"):
    if isinstance(file_name, str):
        file_name = file_name.upper()
    elif not isinstance(file_name, int):
        raise ValueError('Input that allows is a file name(str) or flag(int)!')

    if file_name == 'TRANS' or file_name == 1:
        return SINA_DATA_ROOT + '/trans'
    elif file_name == 'COMM' or file_name == 2:
        return SINA_DATA_ROOT + '/comm'
    else:
        return SINA_DATA_ROOT + '/mblog'


def get_data_filename_via_template(data_type, **kwargs):
    if not isinstance(data_type, str):
        logger.error('str object for "data_type" is allowed. Got %s instead. ' % type(data_type))
    data_type = data_type.lower()

    filename = ''
    if data_type == "sequence":
        try:
            n_users = kwargs['n_users']
            n_samples = kwargs['n_samples']
            filename = USER_DATA_DIR + '/' + SEQ_FILE.format(n_users=n_users, n_samples=n_samples)
        except KeyError as msg:
            logger.error(msg)
    elif data_type == "uid":
        try:
            n_users = kwargs['n_users']
            n_samples = kwargs['n_samples']
            filename = USER_DATA_DIR + '/' + UID_FILE.format(n_users=n_users, n_samples=n_samples)
        except KeyError as msg:
            logger.error(msg)
    elif data_type == 'text':
        try:
            userid = kwargs['userid']
            n_samples = kwargs['n_samples']
            filename = USER_DATA_DIR + '/' + TEXT_FILE.format(userid=userid, n_samples=n_samples)
        except KeyError as msg:
            logger.error(msg)
    elif data_type == 'tfidf':
        try:
            n_users = kwargs['n_users']
            n_samples = kwargs['n_samples']
            filename = USER_DATA_DIR + '/' + TF_IDF_FILE.format(n_users=n_users, n_samples=n_samples)
        except KeyError as msg:
            logger.error(msg)
    else:
        logger.error('Value of "data_type" error. Either of "sequence", "text" or "tfidf" is allowed. ')
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
    print(get_root_dir())












