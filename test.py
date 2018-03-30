import utils.config_util as conf

import numpy as np
import csv
import os

folder_path = conf.RESULT_DIR + '/diff_{n_users}_{n_samples}_{n_dims}'.format(
    n_users=1, n_samples=2, n_dims=3)
for m in range(1):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    with open(folder_path + '/diff_' + str(m) + '.csv', 'w') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow([1,2])
