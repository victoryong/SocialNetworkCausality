# -*- coding: utf-8 -*-

from utils.log import get_console_logger
import utils.config_util as conf
import pymongo
import csv
import re
from datetime import datetime, timedelta

# from gen_data import calculate_pub_time, is_text_invalid
import numpy as np

# print(calculate_pub_time('60分钟前', '2017-08-19 00:59:00'))


# client = pymongo.MongoClient(host='10.21.50.32', port=27017)
# social_db = client['user_social']
# mblog_coll = social_db['Mblog']
# mblogs_csr = mblog_coll.find(projection={'_id': False, 'mblog': True, 'sc_created_time': True},
#                              no_cursor_timeout=True)
# print(mblogs_csr.count())

#
# d = {'1':1,'2':2, '3':2,'4':4,'5':5}
# v = np.array(list(d.values()))
# k = np.array(list(d))[v.argsort()]
# v = v[v.argsort()]
# print(k)
# print(v)
#
# with open(conf.get_root_dir('DATA_ROOT') + '/user_mblog_statistic.csv', 'w', encoding='utf-8') as fp:
#     csv_writer = csv.writer(fp)
#     csv_writer.writerows(zip(k, v))
#     # for item in zip(k, v):
#     #     print(item)
#     #     fp.write(str(item))

d = {'1':[{'m':11},{'m':100}]}
user_mblog_fname = conf.get_root_dir('DATA_ROOT') + '/user_mblogs.csv'
with open(user_mblog_fname, 'w', encoding='utf-8') as fp:
    csv_writer = csv.writer(fp)
    u = 'uid'
    csv_writer.writerow(list(u, ['mid', 'pub_time', 'text']))
    csv_writer.writerow([123,312,100])
