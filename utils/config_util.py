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

# Time format
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
PUB_TIME_FORMAT = '%Y-%m-%d'


# Data directories
ROOT = sys.path[1] if os.name == 'posix' else sys.path[0]
_PP_ROOT = PurePath(ROOT)
PRJ_DATA_ROOT = str(_PP_ROOT.parent) + '/Data/' + str(_PP_ROOT.name)
SINA_DATA_ROOT = str(_PP_ROOT.parent) + '/Data/sina_weibo_data'


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












