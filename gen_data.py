# coding: utf-8
"""
Created on Sun Oct 29 20:36:54 2017

@author: Victor

Read data files and generate input data in the form that is needed below.
"""

import os
import pymongo
from datetime import datetime, timedelta
import re
import numpy as np
import csv
import gc
import time
import threading

import utils.config_util as conf
from utils.log import get_console_logger
from utils.words_segmentation import tokenize
from utils.text_processing import TextProcessor

logger = get_console_logger(__name__)

# Possible forms of time string.
FULL_TIME_FORM = re.compile(r'^\d{4}-\d{2}-\d{2}$')
HALF_TIME_FORM = re.compile(r'^\d{2}-\d{2}')
OFFSET_TIME_FORM = re.compile(r'(\d{1,2})\u5c0f\u65f6\u524d|(\d{1,2})\u5206\u949f\u524d')
# HOUR_OFFSET_TIME_FORM = re.compile(r'(\d{1,2})\u5c0f\u65f6\u524d')
# MINUTE_OFFSET_TIME_FORM = re.compile(r'(\d{1,2})\u5206\u949f\u524d')
YESTERDAY_TIME_FORM = re.compile(r'\u6628\u5929\s\d{2}:\d{2}')
JUST_NOW_TIME_FORM = re.compile(r'^\u521a\u521a$')

# Filters for mblog text.
MBLOG_REPOST = re.compile(r'^\u8f6c\u53d1\u5fae\u535a$|^\u8f49\u767c\u5fae\u535a$|//')
MBLOG_MENTION = re.compile(
    r'<\w+(\s*[\w\-]*=[\'"][\w\:\.\,\?\'\/\+&%\$#\=~_\-@\u4e00-\u9fa5]*[\'"])*\s*>@\S+<\/\w+>:|@\S+\s')
HTML_TAG = re.compile(
    r'<\w+(\s*[\w\-]*=[\'"][\w\:\;\.\,\?\'\/\+&%\$#\=\[\]~_\-@\u4e00-\u9fa5]*[\'"])*\s*>|<\/\w+>|<\w+\/>')
LINK = re.compile(r'http:/{0,2}\S+')
EMOTION = re.compile(r'\[\S*\]|233+')
ENTITY = re.compile(r'&[A-Za-z]+;')
NOT_CHN_ENG_NUM = re.compile(r'[^A-Za-z1-9\u4e00-\u9fa5]|null|NULL')

STOPWORDS_LIST = []


def load_stopwords():
    global STOPWORDS_LIST
    with open(conf.get_absolute_path('lib') + '/stopwords.dat') as fp:
        lines = fp.readlines()
        STOPWORDS_LIST = [x.strip() for x in lines]
        logger.info("Stopwords have been loaded from %s. " % fp.name)
        del lines
        gc.collect()


def content_filter(s, repl=''):
    """
    Find out meaningless parts in string s and replace them with repl.
    :param s: Input str
    :param repl: Replace with this str
    :return: New str after filtering
    """
    s = LINK.sub(repl, s)
    s = MBLOG_REPOST.sub(repl, s)
    s = MBLOG_MENTION.sub(repl, s)
    s = EMOTION.sub(repl, s)
    s = ENTITY.sub(repl, s)
    s = HTML_TAG.sub(repl, s)
    s = NOT_CHN_ENG_NUM.sub(repl, s)
    for word in STOPWORDS_LIST:
        s = s.replace(word, repl)
    return s


# USER_MBLOGS is a dict consists of dicts. Each element has a key of user id and value of user mblogs info.
# It's used when recovering data from mongodb.
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

# Define the operation of statistic
COUNTING = False
SAVING = False


def calculate_pub_time(pub_time, crawl_time):
    crawl_time_dt = datetime.strptime(crawl_time, conf.TIME_FORMAT)
    if len(FULL_TIME_FORM.findall(pub_time)):
        new_pub_time = pub_time
    elif len(HALF_TIME_FORM.findall(pub_time)):
        new_pub_time = '2017-' + pub_time
    elif len(JUST_NOW_TIME_FORM.findall(pub_time)):
        new_pub_time = crawl_time_dt.strftime(conf.PUB_TIME_FORMAT)
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
        # INVALID_COUNT += 1
        return None
    mblog_item = {'mid': mid, 'text': text, 'pub_time': pub_time}
    USER_MBLOGS[uid].append(mblog_item)
    return USER_MBLOGS[uid]


def save_user_mblogs(uids):
    user_mblog_fname = conf.get_absolute_path('data') + '/user_mblogs.csv'
    # All in one
    #
    # with open(user_mblog_fname, 'w', encoding='utf-8') as fp:
    #     csv_writer = csv.writer(fp)
    #     csv_writer.writerow(['uid', 'mid', 'pub_time', 'text'])
    #     # try:
    #     #     while True:
    #     #         item = USER_MBLOGS.popitem()
    #     #         uid = item[0]
    #     #         mblog_list = item[1]
    #     #         for mblog in mblog_list:
    #     #             csv_writer.writerow([uid, mblog['mid'], mblog['pub_time'], mblog['text']])
    #     # except KeyError as e:
    #     #     print(e)
    #     for i in uids:
    #         uid = str(i)
    #         mblog_list = USER_MBLOGS[uid]
    #         for mblog in mblog_list:
    #             csv_writer.writerow([uid, mblog['mid'], mblog['pub_time'], mblog['text']])
    #         logger.info('Mblogs of user "' + uid + '" saved. ')

    #     One in one
    user_mblog_root = conf.get_absolute_path('data') + '/user_mblogs'
    for i in uids:
        uid = str(i)
        try:
            mblog_list = USER_MBLOGS[uid]
        except KeyError:
            logger.error('No mblogs records about user %s' % uid)
            continue
        user_mblog_fname = user_mblog_root + '/' + uid + '-' + str(len(mblog_list)) + '.csv'
        with open(user_mblog_fname, 'w') as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(['uid', 'mid', 'pub_time', 'text'])
            for mblog in mblog_list:
                csv_writer.writerow([uid, mblog['mid'], mblog['pub_time'], mblog['text']])
        logger.info('Mblogs of user "' + uid + '" saved. ')
    logger.info('User mblog counts saved successfully in file. (%s)' % user_mblog_fname)


def add_user_mblog_counts(uid):
    if uid not in USER_MBLOGS_COUNTS:
        USER_MBLOGS_COUNTS[uid] = 0
    USER_MBLOGS_COUNTS[uid] += 1


def save_user_mblog_counts():
    mblog_counts = np.array(list(USER_MBLOGS_COUNTS.values()))
    mblog_ordered_idx = np.argsort(-mblog_counts)
    user_ids = np.array(list(USER_MBLOGS_COUNTS))[mblog_ordered_idx]
    mblog_counts = mblog_counts[mblog_ordered_idx]

    user_mblog_statistic_fname = conf.get_absolute_path('data') + '/user_mblog_statistic.csv'
    with open(user_mblog_statistic_fname, 'w') as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerows(zip(user_ids, mblog_counts))
        logger.info('User mblog counts saved successfully in file. (%s)' % user_mblog_statistic_fname)
    return user_ids, mblog_counts


def statistic_data_from_mongodb(host='127.0.0.1', port=27017, **kwargs):
    client = pymongo.MongoClient(host=host, port=port)
    social_db = client['user_social']
    mblog_coll = social_db['Mblog']
    count = 0
    global COUNTING, SAVING
    COUNTING = kwargs.get('counting', COUNTING)
    SAVING = kwargs.get('saving', SAVING)

    if 'uid_list' in kwargs:
        uids = kwargs['uid_list']
        filter_dict = {'mblog.user.id': {'$in': uids}}
    else:
        uids = []
        filter_dict = {}

    mblogs_csr = mblog_coll.find(filter=filter_dict,
                                 projection={'_id': False, 'mblog': True, 'sc_created_time': True},  # limit=5000000,
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

        text = content_filter(text)
        original_text = content_filter(original_text)

        if COUNTING:
            add_user_mblog_counts(uid)
        if SAVING:
            add_mblog_item(uid, mid, text + original_text, pub_time, crawl_time)

        count += 1
        if not count % 1000:
            print(count)
    # logger.info("Total valid mblog count: %d. Total invalid count: %d. " % (VALID_COUNT, INVALID_COUNT))

    # Order USER_MBLOG_COUNTS and save
    if COUNTING:
        uids, mblog_counts = save_user_mblog_counts()
    # Order and save
    if SAVING:
        save_user_mblogs(uids)


def find_top_ten_mblogs():
    global COUNTING, SAVING
    COUNTING = False
    SAVING = True

    top_ten = list()
    user_mblog_statistic_fname = conf.get_absolute_path('data') + '/user_mblog_statistic.csv'
    with open(user_mblog_statistic_fname, 'r') as fp:
        csv_reader = csv.reader(fp)
        idx = 0
        for item in csv_reader:
            uid = int(item[0])
            top_ten.append(uid)
            print(uid)
            idx += 1
            if idx == 10:
                break
                # if idx <= 10:
                #     continue
                # if idx == 15:
                #     break
    statistic_data_from_mongodb(host='10.21.50.32', uid_list=top_ten)


def find_default_user_mblogs(**kwargs):
    global COUNTING, SAVING
    COUNTING = False
    SAVING = True
    if 'uid_list' in kwargs:
        uid_list = kwargs['uid_list']
    else:
        default_users_filename = conf.get_absolute_path('data') + '/default_users.txt'
        uid_list = []
        with open(default_users_filename, 'r') as fp:
            for line in fp.readlines():
                uid = int(line.split('|')[0])
                print(uid)
                uid_list.append(uid)
    statistic_data_from_mongodb(host='10.21.50.32', uid_list=uid_list)


class DataGenerator:
    def __init__(self, tf_idf_threshold=.5, time_step=24 * 3600):
        self.uidList = []
        self.sequences = []
        self.textList = []
        self.tfIdfWords = []
        self.tfIdfWeights = None
        self.tfIdfThreshold = tf_idf_threshold
        self.timeStep = time_step
        self.seqLen = int(((conf.END_TIME - conf.START_TIME).total_seconds() / self.timeStep))

    def sequence_idx(self, time_str):
        return int(
            ((conf.END_TIME - datetime.strptime(time_str, conf.PUB_TIME_FORMAT)).total_seconds() / self.timeStep))

    def construct_time_series_data(self):
        user_mblogs_dir = conf.get_absolute_path('data') + '/user_mblogs/'
        user_data_dir = conf.get_absolute_path('data') + '/user_data/'
        for filename in os.listdir(user_mblogs_dir):
            try:
                uid = int(filename.split('-')[0])
            except ValueError as msg:
                logger.info('Invalid file name %s' % msg)
                continue

            self.uidList.append(uid)
            absolute_mblogs_fname = user_mblogs_dir + filename

            sequence = [0] * self.seqLen
            text_list = [''] * self.seqLen
            with open(absolute_mblogs_fname, 'r') as fp:
                csv_reader = csv.reader(fp)
                next(csv_reader)
                try:
                    while True:
                        line = next(csv_reader)
                        pub_time = line[2]
                        text = content_filter(line[3]).strip() + ' '
                        idx = self.sequence_idx(pub_time)
                        try:
                            sequence[idx] = 1
                            text_list[idx] += text
                        except IndexError:
                            # logger.info('Index(%d) out of range. Cause: pub_time(%s) is out of range. ' % (idx, pub_time))
                            pass
                except StopIteration:
                    self.sequences.append(sequence)
                    text_list = tokenize(text_list)
                    self.textList.extend(text_list)
                    logger.info('Successfully gen user %d\'s data. ' % uid)
            # Save text a file. One user is related to one file.
            with open(user_data_dir + conf.get_data_filename_via_template('text', userid=uid, n_samples=self.seqLen),
                      'w') as fp:
                csv_writer = csv.writer(fp)
                for row in text_list:
                    csv_writer.writerow([row])
            del text_list
        gc.collect()
        # print self.uidList
        # self.exec_tf_idf()
        # self.save_data()

    def exec_tf_idf(self):
        text_processor = TextProcessor()
        self.tfIdfWords, self.tfIdfWeights = text_processor.tf_idf_transform(self.textList)
        logger.info('TF-IDF processor ended successfully! ')

    def save_data(self):
        uid_fname = conf.get_data_filename_via_template('uid', n_users=len(self.uidList), n_samples=self.seqLen)
        seq_fname = conf.get_data_filename_via_template('seq', n_users=len(self.uidList), n_samples=self.seqLen)
        tf_idf_fname = conf.get_data_filename_via_template('tfidf', n_users=len(self.uidList), n_samples=self.seqLen)
        # Save user ids
        with open(uid_fname, 'w') as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(self.uidList)
            logger.info('User id are saved in %s. ' % uid_fname)
        # Sava sequences
        with open(seq_fname, 'w') as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerows(self.sequences)
            logger.info('Sequences data is saved in file %s. ' % seq_fname)
        # Save tfidf result
        # with open(tf_idf_fname, 'w') as fp:
        #     csv_writer = csv.writer(fp)
        #     csv_writer.writerow(self.tfIdfWords)
        #     csv_writer.writerows(self.tfIdfWeights)
        #     logger.info('TF-IDF data is saved in file %s. ' % tf_idf_fname)


def test():
    user_data_dir = conf.get_absolute_path('data') + '/user_data/'
    test_list = []
    uid_list = []
    for filename in os.listdir(user_data_dir):
        try:
            if filename.split('_')[0] != 'Text':
                continue
            uid = int(filename.split('_')[1])
            uid_list.append(uid)
        except ValueError as msg:
            logger.info('Invalid file name %s' % msg)
            continue
        absolute_mblogs_fname = user_data_dir + filename
        with open(absolute_mblogs_fname, 'r') as fp:
            csv_reader = csv.reader(fp)
            try:
                while True:
                    line = next(csv_reader)
                    # print line
                    text = line[0].strip()
                    test_list.append(text.split())
            except StopIteration:
                logger.info('Successfully recover user %d\'s data. ' % uid)
    print(len(test_list))

    feature_extraction(test_list, len(uid_list), len(test_list)/len(uid_list))


def memory_state(time_long):
    for i in range(time_long):
        mem = conf.get_memory_state()
        if float(mem.split(' ')[2][:-1]) > 20:
            print(mem)
        time.sleep(1)


def feature_extraction(corpus, n_users, n_samples, text_processor=None, untrained=True):
    if not text_processor:
        text_processor = TextProcessor()
        if not untrained:
            text_processor.load_model('tfidf')
            corpus_tf_idf = text_processor.tfIdfModel[corpus]
            corpus_lsi = text_processor.lsiModel[corpus_tf_idf]
        else:
            corpus_tf_idf = text_processor.tf_idf_transform(corpus)
            corpus_lsi = text_processor.lsi_transform(corpus_tf_idf)
    else:
        if untrained:
            corpus_tf_idf = text_processor.tf_idf_transform(corpus)
            corpus_lsi = text_processor.lsi_transform(corpus_tf_idf)
        else:
            corpus_tf_idf = text_processor.tfIdfModel[corpus]
            corpus_lsi = text_processor.lsiModel[corpus_tf_idf]
    # print corpus_lsi
    with open(conf.get_data_filename_via_template('lsi', n_users=n_users, n_samples=n_samples), 'wb') as fp:
        csv_writer = csv.writer(fp)
        for i in corpus_lsi:
            csv_writer.writerow([x[1] for x in i])
        logger.info('Lsi result saved. ')

    return corpus_lsi


if __name__ == '__main__':
    # statistic_data_from_mongodb(host='10.21.50.32')
    # find_top_ten_mblogs()uid_list=[2263978304, 2846253732, 5032225033, 5213225423]
    # find_default_user_mblogs()
    # debug = 0
    debug = 1
    if not debug:
        load_stopwords()
        gen = DataGenerator()
        gen.construct_time_series_data()
    else:
        t = threading.Thread(target=memory_state, args=(30,))
        t.start()
        test()
        t.join()
        # content_filter('')
