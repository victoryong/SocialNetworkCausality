import utils.config_util as conf
from text_processing import TextProcessor

import numpy as np
import csv
import os

a = np.array([[[1,0,0],[0,1,2]], [[0,0,0],[2,2,2]], [[7,8,9],[10,11,12]]])
print(np.concatenate((a[0], a[1]), axis=1))
print(np.concatenate((np.array([]), a[0])))
