import utils.config_util as conf

import numpy as np
import csv
import os

lda_corpus_path = conf.get_data_filename_via_template('lda', n_users=conf.N_USERS, n_samples=conf.N_SAMPLES,
                                                      n_dims=500, postfix='mm')
print(lda_corpus_path)
conf.mk_dir(lda_corpus_path)