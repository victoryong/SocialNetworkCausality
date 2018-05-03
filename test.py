import utils.config_util as conf

import numpy as np
import csv
import os

with open(conf.get_data_filename_via_template('uid', n_users=conf.N_USERS, n_samples=conf.N_SAMPLES)) as fp:
    uid_list = [int(i) for i in fp.readline().split(',')]
for uid in uid_list:
    print(conf.get_data_filename_via_template('text', user_id=uid, n_samples=conf.N_SAMPLES))