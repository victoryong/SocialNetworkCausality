import utils.config_util as conf

import numpy as np
import csv
import os

a = [[1.,22,4,8,9],[3,4,0,1,8]]
b = [[0,2],[3,0]]

a = np.array(a).astype(int)
b = np.array(b)


# print(np.sum(a[:, 0]))
print(a.reshape(10))
