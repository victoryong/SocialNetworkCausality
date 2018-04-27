import utils.config_util as conf

import numpy as np
import csv
import os

print(conf.get_data_filename_via_template('tfidf', n_users=conf.N_USERS,  postfix='mm',
                                          n_samples=conf.N_SAMPLES))
a =[1,2,3]
b = [5,6,7]
print(list(map(lambda m,n: m+n, a,b)))