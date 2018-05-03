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

logger = conf.get_console_logger(__name__)

# 0 for training model, 1 for loading corpus, 2 for loading model, else for nothing.
args = {
    'tfidf': 0,
    'lsi': 0,
    'lda': 0,
    'w2v': 0
}


n_dims_list = [500, 800, 1000]  # A list of n_dims of vectors that the texts are transformed to.
DG = DataGenerator()
TP = TextProcessor()

# Perform tf-idf
tfidf_corpus = None
if args['tfidf'] == 0:
    DG.recover_text_list()  # DG.textList contains all samples of text in which words are split.
    logger.info('Training tfidf model...')
    tfidf_corpus = TP.tf_idf_transform(DG.textList)
elif args['tfidf'] == 1:
    logger.info('Loading tfidf corpus from file...')
    tfidf_corpus = TP.load_corpus('tfidf')
elif args['tfidf'] == 2:
    logger.info('Loading tfidf model from disk...')
    TP.load_model('tfidf')

# Perform lsi
lsi_corpus = None
if args['lsi'] == 0:
    logger.info('Training lsi model...')
    for i in n_dims_list:
        try:
            lsi_corpus = TP.lsi_transform(tfidf_corpus, n_topics=i)
        except MemoryError as e:
            logger.error('%s. %s' % (str(e), conf.get_memory_state()))
        del lsi_corpus
        del TP.lsiModel
        gc.collect()

# Perform lda
lda_corpus = None
if args['lda'] == 0:
    logger.info('Training lda model...')
    for i in n_dims_list:
        try:
            lda_corpus = TP.lda_transform(tfidf_corpus, n_topics=i)
        except MemoryError as e:
            logger.error('%s. %s' % (str(e), conf.get_memory_state()))
        del lda_corpus
        del TP.ldaModel
        gc.collect()

# Perform word2vec
if args['w2v'] == 0:
    if len(DG.textList) == 0:
        DG.recover_text_list()
    del tfidf_corpus
    gc.collect()

    w2v_corpus = None
    logger.info('Training w2v model...')
    for i in n_dims_list:
        w2v_corpus = TP.w2v_transform(DG.textList, n_dims=i)
        del w2v_corpus
        del TP.w2vModel
        gc.collect()





