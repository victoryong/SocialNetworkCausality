# coding: utf-8
"""
Created on Wed April 25 21:38 2018

@author: Xie Yong

Entry of all procedures. Plenty of available operations can be selected and performed.
"""

import numpy as np
import gc

import utils.config_util as conf
from gen_data import DataGenerator
from text_processing import TextProcessor

from methods import testCSE, testMNR, testTE
from te_estimator import te_transform

logger = conf.get_console_logger('Main')

# 1 for training model, 2 for loading corpus, 3 for loading model, else for nothing.
text_model_args = {
    'tfidf': 0,
    'lsi': 0,
    'lda': 0,
    'w2v': 0
}

vec_type_list = ['lda', 'lsi', 'w2v']
n_dims_list = [100, 500, 800]  # A list of n_dims of vectors that the texts are transformed to.
DG = DataGenerator()
TP = TextProcessor()

# Perform tf-idf
tfidf_corpus = None
if text_model_args['tfidf'] == 1:
    DG.recover_text_list()  # DG.textList contains all samples of text in which words are split.
    logger.info('Training tfidf model...')
    tfidf_corpus = TP.tf_idf_transform(DG.textList)
elif text_model_args['tfidf'] == 2:
    logger.info('Loading tfidf corpus from file...')
    tfidf_corpus = TP.load_corpus('tfidf')
elif text_model_args['tfidf'] == 3:
    logger.info('Loading tfidf model from disk...')
    TP.load_model('tfidf')

# Perform lsi
lsi_corpus = None
if text_model_args['lsi'] == 1:
    logger.info('Training lsi model...')
    for i in n_dims_list:
        try:
            lsi_corpus = TP.lsi_transform(tfidf_corpus, n_topics=i)
        except MemoryError as e:
            logger.error('%s. %s' % (str(e), conf.get_memory_state()))
        del lsi_corpus
        gc.collect()

# Perform lda
lda_corpus = None
if text_model_args['lda'] == 1:
    logger.info('Training lda model...')
    for i in n_dims_list:
        try:
            lda_corpus = TP.lda_transform(tfidf_corpus, n_topics=i)
        except MemoryError as e:
            logger.error('%s. %s' % (str(e), conf.get_memory_state()))
        del lda_corpus
        gc.collect()

# Perform word2vec
if text_model_args['w2v'] == 1:
    if len(DG.textList) == 0:
        DG.recover_text_list()
    del tfidf_corpus
    gc.collect()

    w2v_corpus = None
    logger.info('Training w2v model...')
    for i in n_dims_list:
        w2v_corpus = TP.w2v_transform(DG.textList, n_dims=i)
        del w2v_corpus
        gc.collect()


# Perform te, cse, mnr on sequence data to find skeleton.
seq_methods = {
    'mnr': 0,
    'te': 0,
    'cse': 0
}
sequences = None


def load_seq():
    global sequences
    if sequences is None:
        logger.info('Loading sequences...')
        sequences = np.loadtxt(
            conf.get_data_filename_via_template('seq', n_users=conf.N_USERS, n_samples=conf.N_SAMPLES), delimiter=',')
if seq_methods.get('mnr', 0) != 0:
    load_seq()
    logger.info('Performing MNR...')
    testMNR(6, conf.N_SAMPLES, 100, conf.N_USERS, sequences, 0)
if seq_methods.get('te', 0) != 0:
    load_seq()
    logger.info('Performing TE...')
    testTE(6, conf.N_SAMPLES, 100, conf.N_USERS, sequences, 0)
if seq_methods.get('cse', 0) != 0:
    load_seq()
    logger.info('Performing CSE...')
    testCSE(6, conf.N_SAMPLES, 100, conf.N_USERS, sequences, 0)


# Infer causality from text.
text_methods = {
    'te': 1,
    'cse': 0
}

corpus_vec = None
data = []


def load_vec(vec_type, n_dims=conf.N_DIMS):
    global corpus_vec, data
    logger.info('Loading vectors...')
    try:
        corpus_vec = TP.load_corpus(vec_type, n_dims, True)
    except Exception as e:
        raise e
    data = []
    for i in range(conf.N_USERS):
        data.append(corpus_vec[i * conf.N_SAMPLES: (i + 1) * conf.N_SAMPLES])
    data = np.array(data, dtype=np.float)


if text_methods.get('te', 0) != 0:
    for vec_type in vec_type_list:
        for n_dims in n_dims_list:
            try:
                load_vec(vec_type, n_dims=n_dims)
                cn, te_result = te_transform(data, vec_type, n_dims)
                print(te_result)
            except Exception as e:
                logger.error(e)




