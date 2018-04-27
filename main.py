# coding: utf-8
"""
Created on Wed April 25 21:38 2018

@author: Xie Yong

Entry of all procedures. Plenty of available operations can be selected and performed.
"""

import numpy as np

import utils.config_util as conf
from gen_data import DataGenerator
from text_processing import TextProcessor

logger = conf.get_console_logger(__name__)

# 'train' for training model, 'lmodel' for loading model, 'lcorpus' for loading corpus, else for nothing.
target = ['train', 'lcorpus', 'lmodel']
args = {
    'tfidf': 'lcorpus',
    'lsi': 'train',
    'lda': 'train',
    'w2v': 'train'
}


n_dims_list = [500, 800, 1000]  # Define a list of n_dims of vectors that the texts are transformed to.
DG = DataGenerator()
TP = TextProcessor()

# Perform tf-idf
if args['tfidf'] == target[0]:
    DG.recover_text_list()  # DG.textList contains all samples of text in which words are split.
    logger.info('Training tfidf model...')
    TP.tf_idf_transform(DG.textList)
elif args['tfidf'] == target[1]:
    logger.info('Loading tfidf corpus from file...')
    TP.load_corpus('tfidf')
elif args['tfidf'] == target[2]:
    logger.info('Loading tfidf model from disk...')
    TP.load_model('tfidf')

# Perform lsi
lsi_corpus_list = []
lsi_model_list = []
if args['lsi'] == target[0]:
    logger.info('Training lsi model...')
    for i in n_dims_list:
        lsi_corpus_list.append(TP.lsi_transform(TP.tfIdfCorpus, n_topics=i))
        # print(np.array(TP.lsiCorpus).shape)
elif args['lsi'] == target[1]:
    logger.info('Loading lsi corpus from file...')
    for i in n_dims_list:
        try:
            lsi_corpus_list.append(TP.load_corpus('lsi', n_dims=i))
        except Exception as e:
            logger.error('Failed to load lsi corpus with a n_dims of %d! [%s]' % (i, str(e)))
elif args['lsi'] == target[2]:
    logger.info('Loading lsi model from disk...')
    for i in n_dims_list:
        try:
            lsi_model_list.append(TP.load_model('lsi', n_dims=i))
        except Exception as e:
            logger.error('Failed to load lsi model with a n_dims of %d! [%s]' % (i, str(e)))

# Perform lda
lda_corpus_list = []
lda_model_list = []
if args['lda'] == target[0]:
    logger.info('Training lda model...')
    for i in n_dims_list:
        lda_corpus_list.append(TP.lda_transform(TP.tfIdfCorpus, n_topics=i))
elif args['lda'] == target[1]:
    logger.info('Loading lda corpus from file...')
    for i in n_dims_list:
        try:
            lda_corpus_list.append(TP.load_corpus('lda', n_dims=i))
        except Exception as e:
            logger.error('Failed to load lda corpus with a n_dims of %d! [%s]' % (i, str(e)))
elif args['lda'] == target[2]:
    logger.info('Loading lsi model from disk...')
    for i in n_dims_list:
        try:
            lda_model_list.append(TP.load_model('lda', n_dims=i))
        except Exception as e:
            logger.error('Failed to load lda model with a n_dims of %d! [%s]' % (i, str(e)))

# Perform word2vec
if len(DG.textList) == 0:
    DG.recover_text_list()

w2v_corpus_list = []
w2v_model_list = []
if args['w2v'] == target[0]:
    logger.info('Training w2v model...')
    for i in n_dims_list:
        w2v_corpus_list.append(TP.w2v_transform(DG.textList, n_dims=i))
elif args['w2v'] == target[1]:
    logger.info('Loading w2v corpus from file...')
    for i in n_dims_list:
        try:
            w2v_corpus_list.append(TP.load_corpus('w2v', n_dims=i))
        except Exception as e:
            logger.error('Failed to load w2v corpus with a n_dims of %d! [%s]' % (i, str(e)))
elif args['w2v'] == target[2]:
    logger.info('Loading w2v model from disk...')
    for i in n_dims_list:
        try:
            w2v_model_list.append(TP.load_model('w2v', n_dims=i))
        except Exception as e:
            logger.error('Failed to load w2v model with a n_dims of %d! [%s]' % (i, str(e)))





