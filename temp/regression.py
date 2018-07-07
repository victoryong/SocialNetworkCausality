# coding: utf-8
"""
Created on Thu May 15, 2018

@author: Victor Y, Xie

Perform Granger Causality on high dimension data using regressive model.

"""

import numpy as np
from statsmodels.tsa.api import VAR
import statsmodels.api as sm

import pandas


class RegressionMethod:
    def granger_causality(self, data):
        columns = []
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                columns.append(str(i) + str(j))
        # print(columns)

        topic_oriented_data = data[0]
        for i in range(1, len(data)):
            topic_oriented_data = np.concatenate((topic_oriented_data, data[i]), 1)

        topic_oriented_data = pandas.DataFrame(topic_oriented_data, columns=columns)
        print(topic_oriented_data)
        # print(type(topic_oriented_data))


        var_model = VAR(topic_oriented_data)
        results = var_model.fit(2)
        gc_result = results.test_causality(columns, columns, kind='f')
        print(gc_result.summary())


    def test(self):
        mdata = sm.datasets.macrodata.load_pandas().data
        dates = mdata[['year', 'quarter']].astype(int).astype(str)
        quarterly = dates["year"] + "Q" + dates["quarter"]
        from statsmodels.tsa.base.datetools import dates_from_str
        quarterly = dates_from_str(quarterly)
        mdata = mdata[['realgdp', 'realcons', 'realinv']]
        mdata.index = pandas.DatetimeIndex(quarterly)
        data = np.log(mdata).diff().dropna()
        # print(type(data))
        # print(data)

        model = VAR(data)
        results = model.fit(2)
        # print(results.summary())
        gc_result = results.test_causality(['realgdp', 'realcons', 'realinv'], ['realgdp', 'realcons', 'realinv'], kind='f')
        print(gc_result.summary())

if __name__ == '__main__':
    a = np.array([[[1, 21, 0], [0, 1, 2],[11, 30, 10], [10, 51, 22],[1, 10, 0], [20, 1, 62]],
                  [[80, 50, 20], [2, 22, 2],[1, 0, 10], [0, 11, 2],[1, 0, 44], [0, 21, 2]],
                  [[7, 8, 9], [10, 11, 12], [1, 210, 0], [0, 21, 42],[1, 20, 80], [12, 1, 82]]])
    RM = RegressionMethod()
    RM.granger_causality(a)
    # RM.test()


