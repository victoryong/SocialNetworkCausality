# coding: utf-8
"""
Created on Sun Oct 29 20:36:54 2017

@author: Victor Xie

Read data files and generate input data in the form that is needed below.
"""

import csv
import gc
import sys
import re
from datetime import datetime, timedelta

import numpy as np
import pymongo

import utils.config_util as conf
from utils.log import get_console_logger
from words_segmentation import tokenize
from search_timesteps import segment_ts_inner_pro

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


def get_data_from_mongodb(host='127.0.0.1', port=27017, **kwargs):
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
    get_data_from_mongodb(host='10.21.50.32', uid_list=top_ten)


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
    get_data_from_mongodb(host='10.21.50.32', uid_list=uid_list)


class MblogInfo:
    def __init__(self, mblog_type, db_host, db_port, db_name, coll_name, pd_names,
                 start_time, end_time, time_format, time_step):
        self.mblogType, self.dbHost, self.dbPort, self.dbName, self.collName, self.pdNames, \
        self.startTime, self.endTime, self.timeFormat = \
            mblog_type, db_host, db_port, db_name, coll_name, pd_names, \
            datetime.strptime(start_time, time_format), datetime.strptime(end_time, time_format), time_format
        if isinstance(time_step, list):
            # self.timeStep = []
            # time_sum = 0
            # for i in range(len(time_step)):
            #     time_sum += time_step[i]
            #     self.timeStep.append(time_sum)
            self.timeStep = time_step

            total_seconds = (self.endTime - self.startTime).total_seconds()
            l = int(total_seconds / self.timeStep[-1])
            self.seqLen = (l + 1) * len(time_step) if l * self.timeStep[-1] < int(total_seconds) else l * len(time_step)
        else:
            self.timeStep = time_step
            self.seqLen = int(((self.endTime - self.startTime).total_seconds() / self.timeStep))


class DataGenerator:
    def __init__(self, mblog_info):
        self.uidList = []
        self.sequences = []
        self.textList = []
        self.mblogInfo = mblog_info

    def construct_time_series_data(self):
        mblog_info = self.mblogInfo

        def sequence_idx(time_str):
            if isinstance(mblog_info.timeStep, int):
                return int((mblog_info.endTime - datetime.strptime(time_str, mblog_info.timeFormat))
                           .total_seconds() / mblog_info.timeStep)
            else:
                steps_count = len(mblog_info.timeStep)
                total_seconds = (mblog_info.endTime - datetime.strptime(time_str, mblog_info.timeFormat)).total_seconds()
                index = int(total_seconds / mblog_info.timeStep[-1])
                rest = total_seconds - index * mblog_info.timeStep[-1]
                for i in range(steps_count):
                    if rest < mblog_info.timeStep[i]:
                        return index * steps_count + i
                return (index + 1) * steps_count

        with open(conf.get_absolute_path('data') + 'default_users.txt', encoding='utf-8') as fp:
            lines = fp.readlines()

        user_mblogs_dir = conf.get_absolute_path('data') + 'user_mblogs/'

        self.uidList = []
        for line in lines:
            try:
                uid, count = int(line.split('|')[0]), int(line.split('|')[1])
            except ValueError as e:
                logger.error('Invalid uid or count. %s' % e)
                continue
            filename = '{}-{}.csv'.format(uid, count)
            self.uidList.append(uid)
            absolute_mblogs_fname = user_mblogs_dir + filename

            sequence = [0] * mblog_info.seqLen
            text_list = [''] * mblog_info.seqLen
            with open(absolute_mblogs_fname) as fp:
                csv_reader = csv.reader(fp)
                next(csv_reader)
                try:
                    while True:
                        line = next(csv_reader)
                        pub_time = line[2]
                        text = content_filter(line[3]).strip() + ' '
                        idx = sequence_idx(pub_time)
                        try:
                            sequence[idx] = 1
                            text_list[idx] += text
                        except IndexError:
                            # logger.info('Index(%d) out of range. Cause: pub_time(%s) is out of range. ' % (idx, pub_time))
                            print('%d, %d' % (len(sequence), idx))
                except StopIteration:
                    self.sequences.append(sequence)
                    text_list = tokenize(text_list)
                    # self.textList.extend(text_list)
                    logger.info('Successfully gen user %d\'s data. ' % uid)

            # Text of one user is saved to one file.
            with open(conf.get_filename_via_tpl('text', user_id=uid, n_samples=mblog_info.seqLen), 'w') as fp:
                csv_writer = csv.writer(fp)
                for row in text_list:
                    csv_writer.writerow([row])
        # print self.uidList
        self.save_sequences()

    def construct_with_diff_ts(self, new_mblog_info):
        n_users, n_samples = 12, 2192
        uid_list = np.loadtxt(conf.get_filename_via_tpl('uid', n_users=n_users, n_samples=n_samples), delimiter=',', dtype=np.int)
        original_seq = np.loadtxt(conf.get_filename_via_tpl('seq', n_users=n_users, n_samples=n_samples), delimiter=',', dtype=np.int)
        original_text_list = []
        for uid in uid_list:
            with open(conf.get_filename_via_tpl('text', user_id=uid, n_samples=n_samples), encoding='utf-8') as fp:
                csv_reader = csv.reader(fp)
                text = [line[0].strip() if len(line) > 0 else '' for line in csv_reader]
                original_text_list.append(text)
                assert len(text) == 2192, 'Texts of user %d are not enough. ' % uid

        time_steps = new_mblog_info.timeStep
        if n_samples == len(new_mblog_info.timeStep):
            return original_seq, original_text_list

        new_seq = np.zeros((original_seq.shape[0], len(time_steps)), np.int)
        new_text_list = [[''] * len(time_steps)] * original_seq.shape[0]

        nidx = oidx = 0
        for time_point in time_steps:
            step = int(time_point / (24 * 3600) - oidx)
            # if step == 1:
            #     new_seq[:, nidx] = original_seq[:, oidx]
            #     new_text_list[:, nidx] = original_text_list[:, oidx]
            # else:
            for r in range(original_seq.shape[0]):
                new_seq[r, nidx] = sum(original_seq[r, oidx: oidx + step])
                new_text_list[r][nidx] = original_text_list[r][oidx]
                for c in range(1, step):
                    new_text_list[r][nidx] = new_text_list[r][nidx] + ' ' + original_text_list[r][oidx + c]
                    # print(original_text_list[r][oidx + c])
            new_seq[new_seq[:, nidx] > 0, nidx] = 1

            oidx += step
            nidx += 1
        assert nidx == len(time_steps), 'nidx != len(time_steps)'
        assert oidx == original_seq.shape[1], 'Total amount of time steps is smaller than sample length.'

        self.uidList = uid_list
        self.sequences = new_seq
        self.save_sequences(new_mblog_info)

        for uid, texts in zip(uid_list, new_text_list):
            with open(conf.get_filename_via_tpl('text', user_id=uid, n_samples=new_mblog_info.seqLen), 'w') as fp:
                csv_writer = csv.writer(fp)
                for row in texts:
                    csv_writer.writerow([row])

        return new_seq, new_text_list

    def save_sequences(self, new_mblog_info=None):
        mblog_info = new_mblog_info if new_mblog_info else self.mblogInfo
        uid_fname = conf.get_filename_via_tpl('uid', n_users=len(self.uidList), n_samples=mblog_info.seqLen)
        seq_fname = conf.get_filename_via_tpl('seq', n_users=len(self.uidList), n_samples=mblog_info.seqLen)
        # Save user ids
        with open(uid_fname, 'w') as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(self.uidList)
            logger.info('User id are saved in %s. ' % uid_fname)
        # Save sequences
        with open(seq_fname, 'w') as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerows(self.sequences)
            logger.info('Sequences data is saved in file %s. ' % seq_fname)


def recover_text_list(n_users, n_samples, debug=False):
    text_list = []

    with open(conf.get_filename_via_tpl('uid', n_users=n_users, n_samples=n_samples)) as fp:
        uid_list = [int(i) for i in fp.readline().split(',')]

    debug_flag = 0
    for uid in uid_list:
        if debug and debug_flag > 0:
            break

        csv.field_size_limit(sys.maxsize)
        with open(conf.get_filename_via_tpl('text', user_id=uid, n_samples=n_samples), encoding='utf-8') as fp:
            csv_reader = csv.reader(fp)
            for line in csv_reader:
                if not len(line) or line[0] == '':
                    text = []
                else:
                    text = line[0].strip().split(' ')
                    while '' in text:
                        text.remove('')
                text_list.append(text)
            logger.info('Successfully recover user %d\'s data. ' % uid)
        debug_flag += 1
    # for i in range(10):
    #     print(text_list[i])
    if debug:
        print(text_list[:100])
    return text_list


if __name__ == '__main__':
    # Segment data with a constant time step 1 day which is also the smallest duration.
    mblog_info = MblogInfo('mblog', '10.21.50.32', 27017, 'user_social', 'Mblog', [],
                           '2011-10-01', '2017-10-01', '%Y-%m-%d', 24*3600)
    DG = DataGenerator(mblog_info)

    # get_data_from_mongodb(host='10.21.50.32')
    # find_top_ten_mblogs()uid_list=[2263978304, 2846253732, 5032225033, 5213225423]
    # find_default_user_mblogs()

    # First time construct ts data.
    # DG.construct_time_series_data()

    ts_search_result = segment_ts_inner_pro()[1:]
    ts_search_result = np.array(ts_search_result)

    last_len = -1
    for ts in ts_search_result:
        new_mblog_info = MblogInfo('mblog', '10.21.50.32', 27017, 'user_social', 'Mblog', [],
                                   '2011-10-01', '2017-09-30', '%Y-%m-%d', (24 * 3600 * ts[0]).tolist())
        print(len(ts[0]))
        if last_len == len(ts[0]):
            continue
        last_len = len(ts[0])
        DG.construct_with_diff_ts(new_mblog_info)

    # DG.recover_text_list(True)

