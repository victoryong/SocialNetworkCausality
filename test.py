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
# mblogs_csr = mblog_coll.find(filter={'mblog.user.id': 2187909461}, projection={'_id': False, 'mblog.user.id': True, 'sc_created_time': True},
#                              no_cursor_timeout=True)
# print(mblogs_csr.count())
# for i in mblogs_csr:'mblog.user.id':{'$in': [2919618582, 1705570705]}
#     print(i)

#
# d = {'1':1,'2':2, '3':2,'4':4,'5':5}
# v = np.array(list(d.values()))
# k = np.array(list(d))[v.argsort()]
# v = v[v.argsort()]
# print(k)
# print(v)
#
# with open(conf.get_root_dir('DATA_ROOT') + '/user_mblog_statistic00.csv', 'w', encoding='utf-8') as fp:
#     csv_writer = csv.writer(fp)
#     csv_writer.writerows(zip(k, v))


# user_mblog_statistic_fname = conf.get_root_dir('DATA_ROOT') + '/user_mblog_statistic.csv'
# with open(user_mblog_statistic_fname, 'r', encoding='utf-8') as fp:
#     csv_reader = csv.reader(fp)

# s = ''' #王源# #王源青春旅社# 感谢分享<img src="h5.sinaimg.cn/m/emoticon/icon/others/l_xin-8e9a1a0346.png" style="width:1em;height:1em;" alt="[心]">23号晚十点约定你哦<img src="h5.sinaimg.cn/m/emoticon/icon/minions/minions_gaoxing-fa40ea5822.png" style="width:1em;height:1em;" alt="[小黄人高兴]">@TFBOYS-王源  weibo.com/p/10151501_62720222   weibo.com/p/10151501_100313915 <a class='k' href='https://m.weibo.cn/k/%E9%9D%92%E6%98%A5%E6%97%85%E7%A4%BE?from=feed'>#青春旅社#</a> <a class='k' href='https://m.weibo.cn/k/%E7%8E%8B%E6%BA%90%E9%9D%92%E6%98%A5%E6%97%85%E7%A4%BE?from=feed'>#王源青春旅社#</a> <a href='https://m.weibo.cn/n/TFBOYS-王源'>@TFBOYS-王源</a> 跟着哥哥姐姐们一起体验创业者的艰辛，感受过往旅客的质朴与纯真，回归最美好的青春时光，愿你永远做个自在随风的少年！想想都是美好的<span class="url-icon"><img src="//h5.sinaimg.cn/m/emoticon/icon/others/l_xin-8e9a1a0346.png" style="width:1em;height:1em;" alt="[心]"></span>9月23日晚10点第0集腾讯视频全网首播！<span class="url-icon"><img src="//h5.sinaimg.cn/m/emoticon/icon/default/d_aini-2b6c9354c7.png" style="width:1em;height:1em;" alt="[爱你]"></span> ​​​'''
# MBLOG_REPOST = re.compile(r'^\u8f6c\u53d1\u5fae\u535a$|^\u8f49\u767c\u5fae\u535a$|//')
# MBLOG_MENTION = re.compile(r'<\w+(\s*[\w\-]*=[\'"][\w\:\.\,\?\'\/\+&%\$#\=~_\-@\u4e00-\u9fa5]*[\'"])*\s*>@\S+<\/\w+>:')
# HTML_TAG = re.compile(r'<\w+(\s*[\w\-]*=[\'"][\w\:\;\.\,\?\'\/\+&%\$#\=\[\]~_\-@\u4e00-\u9fa5]*[\'"])*\s*>|<\/\w+>')
# s = MBLOG_REPOST.sub('', s)
# s = MBLOG_MENTION.sub('', s)
# s = HTML_TAG.sub('', s)
# print(s)

# import utils.config_util as conf
# d = datetime.strptime('2017-10-02', conf.PUB_TIME_FORMAT)
# dd = datetime.strptime('2017-10-29', conf.PUB_TIME_FORMAT)
# print(int(((conf.END_TIME-conf.START_TIME).total_seconds()/86400)))
#
# from collections import Iterable
# a = [1,2]
# ia = iter(a)
# try:
#     while True:
#         print(next(ia))
# except StopIteration as msg:
#     print('oh no %s' % msg)

# a = ['123']
# b = ['fdas', 'sdaf']
# a.extend(b)
# print(a)
# a = ['1231132', 'sadf']
# with open(conf.get_absolute_path('DATA_ROOT') + '/temp.csv', 'w') as fp:
#     w = csv.writer(fp)
#     for row in a:
#         w.writerow([row])
import csv
from gen_data import content_filter
with open(conf.get_absolute_path('data_root') + '/user_data/Text_1194765363_2192.csv', 'r') as fp:
    print(fp.readline())
    r = csv.reader(fp)
    print(content_filter(next(r)[0].strip()))


