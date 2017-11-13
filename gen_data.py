# coding: utf-8
"""
Created on Sun Oct 29 20:36:54 2017

@author: Xie Yong

Read data files and generate input data in the form that is needed below.
"""

import os
import pymongo
from datetime import datetime, timedelta
import re
import numpy as np
import csv

import utils.config_util as conf
from utils.log import get_console_logger

logger = get_console_logger(__name__)

START_TIME = datetime.strptime('2014-01-01 00:00:00', conf.TIME_FORMAT)
END_TIME = datetime.strptime('2017-07-01 00:00:00', conf.TIME_FORMAT)

FULL_TIME_FORM = re.compile(r'^\d{4}-\d{2}-\d{2}$')
HALF_TIME_FORM = re.compile(r'^\d{2}-\d{2}')
OFFSET_TIME_FORM = re.compile(r'(\d{1,2})\u5c0f\u65f6\u524d|(\d{1,2})\u5206\u949f\u524d')
# HOUR_OFFSET_TIME_FORM = re.compile(r'(\d{1,2})\u5c0f\u65f6\u524d')
# MINUTE_OFFSET_TIME_FORM = re.compile(r'(\d{1,2})\u5206\u949f\u524d')
YESTERDAY_TIME_FORM = re.compile(r'\u6628\u5929\s\d{2}:\d{2}')

# USER_MBLOGS is a dict consists of dicts. Each element has a key of user id and value of user mblogs info.
#
# USER_MBLOGS = {
#     "123456789":[{
#         "mid": "987654321",
#         "text": "user's text" + " " + "original text",
#         "pub_time": "2017-01-01",
#         "crawl_time": "2017-09-01"
#     },{
#         ...
#     }],
#     "...": [{...}, ...],
# }
USER_MBLOGS = dict()
USER_MBLOGS_COUNTS = dict()
VALID_COUNT = 0
INVALID_COUNT = 0


def calculate_pub_time(pub_time, crawl_time):
    crawl_time_dt = datetime.strptime(crawl_time, conf.TIME_FORMAT)
    if len(FULL_TIME_FORM.findall(pub_time)):
        new_pub_time = pub_time
    elif len(HALF_TIME_FORM.findall(pub_time)):
        new_pub_time = '2017-' + pub_time
    elif len(YESTERDAY_TIME_FORM.findall(pub_time)):
        new_pub_time = (crawl_time_dt - timedelta(1)).strftime(conf.PUB_TIME_FORMAT)
    else:
        offset = OFFSET_TIME_FORM.findall(pub_time)
        if len(offset):
            try:
                hours = int(offset[0][0])
            except ValueError:
                hours = 0
            try:
                minutes = int(offset[0][1])
            except ValueError:
                minutes = 0
            new_pub_time = (crawl_time_dt - timedelta(hours=hours, minutes=minutes)).strftime(conf.PUB_TIME_FORMAT)
        else:
            logger.error('Error: Wrong pub_time string. %s' % pub_time)
            new_pub_time = pub_time
    return new_pub_time


def is_text_invalid(text):
    # 无/抱歉，您...
    if len(re.findall(r'^\u65e0$|^\u62b1\u6b49\uff0c\u60a8', text)):
        return True
    return False


def add_mblog_item(uid, mid, text, pub_time, crawl_time):
    if uid not in USER_MBLOGS:
        USER_MBLOGS[uid] = []
    pub_time = calculate_pub_time(pub_time, crawl_time)
    if is_text_invalid(text):
        INVALID_COUNT += 1
        return None
    mblog_item = {'mid': mid, 'text': text, 'pub_time': pub_time}
    USER_MBLOGS[uid].append(mblog_item)
    return USER_MBLOGS[uid]


def add_user_mblog_counts(uid):
    if uid not in USER_MBLOGS_COUNTS:
        USER_MBLOGS_COUNTS[uid] = 0
    USER_MBLOGS_COUNTS[uid] += 1


def statistic_data_from_mongodb(host='127.0.0.1', port=27017):
    client = pymongo.MongoClient(host=host, port=port)
    social_db = client['user_social']
    mblog_coll = social_db['Mblog']

    mblogs_csr = mblog_coll.find(projection={'_id': False, 'mblog': True, 'sc_created_time': True}, limit=500000,
                                 no_cursor_timeout=True)
    for mblog in mblogs_csr:
        try:
            mblog_info = mblog['mblog']
            uid = str(mblog_info['user']['id'])
            mid = mblog_info['idstr']
            text = mblog_info['text']
            pub_time = mblog_info['created_at']
        except KeyError as err:
            logger.error('Error: Mblog information missing! Skip this item. ' + str(err))
            continue
        crawl_time = mblog.get("sc_created_time", "2000-01-01")
        try:
            original_text = ' ' + mblog_info['retweeted_status']['text']
        except KeyError:
            original_text = ''
        print(uid + '-' + mid + ': ' + text)
        add_user_mblog_counts(uid)
        add_mblog_item(uid, mid, text+original_text, pub_time, crawl_time)

        # try:
        #     s = str(mblog['mblog']['created_at'])
        #     if not len(FULL_TIME_FORM.findall(s)) and not len(HALF_TIME_FORM.findall(s)) and \
        #             not len(OFFSET_TIME_FORM.findall(s)):
        #         print(mblog)
        # except KeyError:
        #     pass
    logger.info("Total valid mblog count: %d. Total invalid count: %d. " % (VALID_COUNT, INVALID_COUNT))

    # Order USER_MBLOG_COUNTS
    mblog_counts = np.array(list(USER_MBLOGS_COUNTS.values()))
    mblog_ordered_idx = mblog_counts.argsort()
    user_ids = np.array(list(USER_MBLOGS_COUNTS))[mblog_ordered_idx]
    mblog_counts = mblog_counts[mblog_ordered_idx]
    user_mblog_statistic_fname = conf.get_root_dir('DATA_ROOT') + '/user_mblog_statistic3.csv'
    with open(user_mblog_statistic_fname, 'w', encoding='utf-8') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerows(zip(user_ids, mblog_counts))
        logger.info('User mblog counts saved successfully in file. (%s)' % user_mblog_statistic_fname)


if __name__ == '__main__':
    statistic_data_from_mongodb(host='10.21.50.32')
    # print(USER_MBLOGS)
