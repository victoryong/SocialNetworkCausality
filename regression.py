# coding: utf-8
"""
Created on Thu May 15, 2018

@author: Victor Y, Xie

Perform Granger Causality on high dimension data using regressive model.

"""

import sklearn.linear_model as lm
import numpy as np
from statsmodels.tsa


class RegressionMethod:
    def var(self, data):
        lasso_model = lm.Lasso(alpha=0.1)
        topics_samples = data[0]
        for i in range(1, len(data)):
            topics_samples = np.concatenate((topics_samples, data[i]), axis=1)
        # print(topics_samples)


    def group_lasso(self):
        pass



if __name__ == '__main__':
    a = np.array([[[1, 0, 0], [0, 1, 2]], [[0, 0, 0], [2, 2, 2]], [[7, 8, 9], [10, 11, 12]]])
    RM = RegressionMethod()
    RM.var(a)



