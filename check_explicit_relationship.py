# coding: utf-8
"""
Created on 15 Nov Wed 2017

@author: Victor Y, Xie

Check relationships such as retweets and comments of users. These relationships are explicit.
"""

import pymongo
import csv
import numpy as np
import os

from utils import log
import utils.config_util as conf

logger = log.get_console_logger(__name__)
# Connect to db
HOST = '10.21.50.32'
PORT = 27017
COLL_NAME = 'Mblog'
CLIENT = pymongo.MongoClient(host=HOST, port=PORT)
SOCIAL_DB = CLIENT['user_social']
COLLECTION = SOCIAL_DB['Mblog']


# def update_db_connection(**kwargs):
#     global HOST, PORT, COLL_NAME, CLIENT, SOCIAL_DB, COLLECTION
#     if 'host' in :
#         CLIENT = pymongo.MongoClient(host=kwargs['host'], port=kwargs['port'])
#         SOCIAL_DB = CLIENT['user_social']
#
#     if kwargs.get('db_name', COLL_NAME) != COLL_NAME:
#         COLLECTION


def check_a_pair_of_user(user_pair):
    logger.info("Checking out if user(%d) had retweeted user(%d)'s mblogs... " % (user_pair[0], user_pair[1]))
    return COLLECTION.find(filter={'mblog.user.id': int(user_pair[0]),
                                   'mblog.retweeted_status.user.id': int(user_pair[1])}).count()


def check_retweet(user_pairs=None, **kwargs):
    """
    Check if each pair of users had retweeted relationship. Each pair is a tuple.
    :param user_pairs: A tuple or list of tuples.
    :return: If user_pairs is a tuple tp, return number indicates that tp[0] had retweeted tp[1]'s tweet. Particularly,
    None represents error input. Return a list of integer when user_pairs is a list of tuples.
    """
    if not user_pairs:
        logger.info('No user pair input! ')
        return

    # update_db_connection(kwargs)
    if isinstance(user_pairs, tuple):
        return check_a_pair_of_user(user_pairs)
    if isinstance(user_pairs, list):
        has_retweeted = []
        for user_pair in user_pairs:
            if not isinstance(user_pair, tuple):
                logger.error('User pair must be a tuple! Got %s(%s) instead. ' % (user_pair, type(tp)))
                has_retweeted.append(None)
                continue
            has_retweeted.append(check_a_pair_of_user(user_pair))
        return has_retweeted


def find_retweet_users(uid):
    logger.info('Finding users who had retweeted user(%d)\'s mblogs...' % uid)
    retweetded_csr = COLLECTION.find(filter={'mblog.retweeted_status.user.id': uid},
                                     projection={'_id': False, 'mblog.user.id': True})
    retweeted_fname = conf.get_absolute_path('DATA_ROOT') + '/retweeted/' + str(uid) + '.csv'

    retweeted_counts = {}
    for user in retweetded_csr:
        uid = user['mblog']['user']['id']
        if uid not in retweeted_counts:
            retweeted_counts[uid] = 1
        else:
            retweeted_counts[uid] += 1

    counts_list = np.array(list(retweeted_counts.values()))
    indices_list = counts_list.argsort()
    uid_list = np.array(list(retweeted_counts))[indices_list]
    counts_list = counts_list[indices_list]
    with open(retweeted_fname, 'w', encoding='utf-8') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerows(zip(uid_list, counts_list))
        logger.info('Saving user id and retweet counts to file(%s) successfully! ' % retweeted_fname)


if __name__ == '__main__':
    # print(check_retweet([
    #     (2920061512, 2919618582),
    #     (1194765363, 5405842841),
    #     (2920061512, 5405842841),
    # ], host='10.21.50.32'))
    # find_retweet_users(1661667591)

    # top_ten = list()
    # user_mblog_statistic_fname = conf.get_root_dir('DATA_ROOT') + '/user_mblog_statistic.csv'
    # with open(user_mblog_statistic_fname, 'r', encoding='utf-8') as fp:
    #     csv_reader = csv.reader(fp)
    #     idx = 0
    #     for item in csv_reader:
    #         uid = int(item[0])
    #         top_ten.append(uid)
    #         print(uid)
    #         idx += 1
    #         if idx == 10:
    #             break

    # Find retweet data of users whose mblogs are collected.
    # user_mblogs_dir = conf.get_root_dir('DATA_ROOT') + '/user_mblogs/'
    # for filename in os.listdir(user_mblogs_dir):
    #     try:
    #         uid = int(filename.split('-')[0])
    #         find_retweet_users(uid)
    #     except ValueError as msg:
    #         logger.error('Not a valid file. Skip it. ' + str(msg))
    uid_list = []
    user_mblogs_dir = conf.get_absolute_path('DATA_ROOT') + '/user_mblogs/'
    for filename in os.listdir(user_mblogs_dir):
        try:
            uid = int(filename.split('-')[0])
            uid_list.append(uid)
        except ValueError as msg:
            logger.error('Not a valid file. Skip it. ' + str(msg))
    uid_pairs = []
    for uid in uid_list:
        for uid_2 in uid_list:
            if uid != uid_2:
                uid_pairs.append((uid, uid_2))
    users_retweet = check_retweet(uid_pairs)
    with open(conf.get_absolute_path('DATA_ROOT') + '/users_retweet.csv', 'w', encoding='utf-8') as fp:
        csv_writer = csv.writer(fp)
        for idx in range(len(uid_pairs)):
            csv_writer.writerow([str(uid_pairs[idx][0]) + '-->' + str(uid_pairs[idx][1]), str(users_retweet[idx])])
