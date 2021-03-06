# -*- coding: utf-8 -*-
"""
Create on Nov 22 Sat 2017

@author: Victor Xie

A text processor whose aims at vectorizing text, includes tf-idf, lsi, lda, word2vec, etc.
"""

import os
import csv

import numpy as np
from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel
from gensim.models.word2vec import Word2Vec
from gensim import corpora, matutils

import utils.config_util as conf
from utils.log import get_console_logger
from gen_data import recover_text_list
from optimize_obj_func import get_n_samples_list

logger = get_console_logger(__name__)


class TextProcessor:
    def __init__(self, n_users, n_samples, n_dims):
        self.nUsers, self.nSamples, self.nDims = n_users, n_samples, n_dims
        self.tfIdfModel = self.lsiModel = self.ldaModel = self.w2vModel = self.dictionary = None

        self.dictPath, self.tfIdfPath, self.lsiPath, self.ldaPath, self.w2vPath, self.w2vVecPath =\
            conf.get_filename_via_tpl('model', model_type='tfidf', n_users=n_users, n_samples=n_samples, model_filename='dict'), \
            conf.get_filename_via_tpl('model', model_type='tfidf', n_users=n_users, n_samples=n_samples, model_filename='tfidf'),\
            conf.get_filename_via_tpl('model', model_type='lsi', n_users=n_users, n_samples=n_samples, n_dims=n_dims, model_filename='lsi_model'), \
            conf.get_filename_via_tpl('model', model_type='lda', n_users=n_users, n_samples=n_samples, n_dims=n_dims, model_filename='lda_model'),\
            conf.get_filename_via_tpl('model', model_type='w2v', n_users=n_users, n_samples=n_samples, n_dims=n_dims, model_filename='w2vmodel'), \
            conf.get_filename_via_tpl('model', model_type='w2v', n_users=n_users, n_samples=n_samples, n_dims=n_dims, model_filename='vec.txt')

    def load_model(self, model_type):
        model = None
        try:
            if model_type == 'tfidf':
                model = TfidfModel.load(self.tfIdfPath, mmap='r')
                self.tfIdfModel = model
            elif model_type == 'lsi':
                model = LsiModel.load(self.lsiPath, mmap='r')
                self.lsiModel = model
            elif model_type == 'lda':
                model = LdaModel.load(self.ldaPath, mmap='r')
                self.ldaModel = model
            elif model_type == 'w2v':
                model = Word2Vec.load(self.w2vPath, mmap='r')
                self.w2vModel = model
            else:
                logger.error('Model type error. Unexpected %s' % model_type)
                return None

            if self.dictionary is None and os.path.exists(self.dictPath):
                self.dictionary = corpora.Dictionary.load(self.dictPath)

            logger.info('%s model loaded completely.' % model_type)
        except IOError:
            logger.error('The %s model doesn\'t exist. Please train the model before load it.' % model_type)
        finally:
            return model

    def tf_idf_transform(self, doc):
        """
        Perform tf-idf transformation on doc.
        """
        self.dictionary = corpora.Dictionary(doc)
        corpus = [self.dictionary.doc2bow(text) for text in doc]
        self.tfIdfModel = TfidfModel(corpus)

        conf.mk_dir(self.tfIdfPath)

        self.dictionary.save(self.dictPath)
        logger.info('Dictionary has been saved in %s.' % self.dictPath)

        self.tfIdfModel.save(self.tfIdfPath)
        logger.info('TF-IDF model has been saved in %s.' % self.tfIdfPath)

        tfidf_corpus = self.tfIdfModel[corpus]
        tfidf_corpus_path = conf.get_filename_via_tpl('tfidf', n_users=self.nUsers, postfix='mm', n_samples=self.nSamples)
        corpora.MmCorpus.serialize(tfidf_corpus_path, tfidf_corpus)
        logger.info('TF-IDF corpus with a shape of %s has been saved in %s.' %
                    (np.array(tfidf_corpus).shape, tfidf_corpus_path))

        return tfidf_corpus

    def lsi_transform(self, corpus_tf_idf):
        logger.info('Training lsi model with a n_dims of %d...' % self.nDims)
        if self.dictionary is None and os.path.exists(self.dictPath):
            self.dictionary = corpora.Dictionary.load(self.dictPath)

        self.lsiModel = LsiModel(corpus=corpus_tf_idf, num_topics=self.nDims, id2word=self.dictionary)
        # print self.lsiModel[corpus]

        conf.mk_dir(self.lsiPath)

        self.lsiModel.save(self.lsiPath)
        logger.info('Lsi model has been saved in %s.' % self.lsiPath)

        lsi_corpus = self.lsiModel[corpus_tf_idf]
        lsi_corpus_path = conf.get_filename_via_tpl('lsi', n_users=self.nUsers, n_samples=self.nSamples,
                                                    n_dims=self.nDims, postfix='mm')
        conf.mk_dir(lsi_corpus_path)
        corpora.MmCorpus.serialize(lsi_corpus_path, lsi_corpus)
        logger.info(
            'Lsi corpus with a shape of %s has been saved in %s.' % (np.array(lsi_corpus).shape, lsi_corpus_path))

        return lsi_corpus

    def lda_transform(self, corpus_tf_idf, train_separated=False, is_update=False):
        """
        Init a lda model with a n_topics whose default is 500, then fit it with corpus_tf_idf and transform it.
        :param corpus_tf_idf: Corpus which has been transformed into tf-idf matrix.
        :param train_separated: The model is going to be train with all corpus one time or some of them separately one time.
        :param is_update: Whether the training to be perform is to construct a new model or update one existed.
        :return: lda corpus.
        """
        logger.info('Training lda model with a n_dims of %d...' % self.nDims)
        if self.dictionary is None and os.path.exists(self.dictPath):
            self.dictionary = corpora.Dictionary.load(self.dictPath)

        if is_update:
            # A ldaModel had been trained before and now update the model with other corpus.
            if self.ldaModel is None:
                self.load_model('lda')
            self.ldaModel.update(corpus_tf_idf)
            logger.info('Lda model has been updated successfully.')
            return self.ldaModel[corpus_tf_idf]

        if train_separated:
            # corpus = []
            # spacing = 10000
            # for i in range(int(len(corpus_tf_idf)/spacing)):
            #     corpus.append(corpus_tf_idf[i*spacing: i])
            # self.ldaModel = LdaModel()
            pass

        self.ldaModel = LdaModel(corpus=corpus_tf_idf, num_topics=self.nDims, id2word=self.dictionary)

        conf.mk_dir(self.ldaPath)
        self.ldaModel.save(self.ldaPath)
        logger.info('lda model has been saved in %s' % self.ldaPath)

        lda_corpus = self.ldaModel[corpus_tf_idf]
        lda_corpus_path = conf.get_filename_via_tpl('lda', n_users=self.nUsers, n_samples=self.nSamples,
                                                    n_dims=self.nDims, postfix='mm')
        conf.mk_dir(lda_corpus_path)
        corpora.MmCorpus.serialize(lda_corpus_path, lda_corpus)
        logger.info(
            'Lda corpus with a shape of %s has been saved in %s.' % (np.array(lda_corpus).shape, lda_corpus_path))

        return lda_corpus

    def w2v_transform(self, sentences):
        """
        Perform word2vec on texts and obtain a w2v model.
        :param sentences: Sentences that each one of it contains a list of words of a text.
        :return: W2v model.
        """
        logger.info('Training w2v model with a dim of %d...' % self.nDims)
        # file = open(infile_path, 'r', encoding='utf-8') if infile_path.find('\n') < 0 else StringIO(infile_path)
        # sentences = []
        # for sen in file.readlines():
        #     sentences.append(sen.strip().split(' '))
        # print(sentences)
        self.w2vModel = Word2Vec(sentences, size=self.nDims, min_count=0)

        conf.mk_dir(self.w2vPath)
        self.w2vModel.save(self.w2vPath)
        self.w2vModel.wv.save_word2vec_format(self.w2vVecPath, binary=False)
        # print(model['['])

        # Construct w2v corpus
        w2v_corpus = []
        for sen in sentences:
            vec = [0] * self.nDims
            if len(sen) > 0:
                for word in sen:
                    vec = list(map(lambda m, n: m + n, vec, self.w2vModel[word]))
                    # vec += self.w2vModel[word]
            w2v_corpus.append(vec)

        w2v_corpus_path = conf.get_filename_via_tpl('w2v', n_users=self.nUsers, n_samples=self.nSamples, n_dims=self.nDims)
        conf.mk_dir(w2v_corpus_path)

        with open(w2v_corpus_path, 'w') as fp:
            csv_writer = csv.writer(fp)
            for line in w2v_corpus:
                csv_writer.writerow(line)
        logger.info('W2v corpus has been saved in %s. ' % w2v_corpus_path)

        return w2v_corpus

    def load_corpus(self, model_type, dense=False):
        corpus = None
        try:
            if model_type == 'tfidf':
                corpus = corpora.MmCorpus(
                    conf.get_filename_via_tpl('tfidf', n_users=self.nUsers, postfix='mm', n_samples=self.nSamples))
            elif model_type in ['lsi', 'lda']:
                corpus = corpora.MmCorpus(conf.get_filename_via_tpl(
                    model_type, n_users=self.nUsers, n_samples=self.nSamples, n_dims=self.nDims, postfix='mm'))
            elif model_type == 'w2v':
                corpus = np.loadtxt(conf.get_filename_via_tpl(
                    model_type, n_users=self.nUsers, n_samples=self.nSamples, n_dims=self.nDims), dtype=np.float,
                    delimiter=',')

            logger.info('%s corpus with a shape of %s has been loaded. ' % (model_type, np.array(corpus).shape))

            if dense and model_type in ['tfidf', 'lsi', 'lda']:
                corpus = matutils.corpus2dense(corpus, self.nDims, self.nSamples * self.nUsers, dtype=np.float).T
            else:
                corpus = np.array(corpus)
        except Exception as e:
            raise e
        return corpus

    @staticmethod
    def corpus2dense(corpus, n_terms, n_docs=conf.N_SAMPLES, dtype=np.float):
        return matutils.corpus2dense(corpus, n_terms, n_docs, dtype).T

    def load_vec(self, vec_type):
        logger.info('Loading %s vectors...' % vec_type)
        try:
            corpus_vec = self.load_corpus(vec_type, True)
        except Exception as e:
            raise e
        data = []
        for i in range(self.nUsers):
            data.append(corpus_vec[i * self.nSamples: (i + 1) * self.nSamples])
        data = np.array(data, dtype=np.float)
        return data

def _test():
    tp = TextProcessor(10, 2192, 100)
    corpus = ['你好 中国', '打印 每类 文本 for', 'for', '遍历 所有 文本 第二个 for 便利 某一类 文本 下 的 词语 权重']
    print(tp.w2v_transform('\n'.join(corpus))['你好'])
    corpus = ['你好 中国'.split(), '打印 每类 文本 for'.split(), 'for'.split(), '遍历 所有 文本 第二个 for 便利 某一类 文本 下 的 词语 权重'.split()]
    ctf = tp.tf_idf_transform(corpus)
    clsi = tp.lda_transform(ctf)

    for i in clsi:
        print(i)
    aa = tp.ldaModel.print_topics(num_topics=500, num_words=50)
    for i in aa:
        print(i)

    path = conf.get_filename_via_tpl(
            'model',
            model_type='lsi',
            n_users=conf.N_USERS,
            n_samples=conf.N_SAMPLES,
            n_dims=conf.N_DIMS)
    tp.load_model('lsi')
    tp.w2v_transform([['你好啊', 'hell0'], ['123', 'forfor']])

if __name__ == '__main__':
    n_users = 12
    n_samples_list = get_n_samples_list()

    for n_samples in n_samples_list:
        data_info = {'n_users': n_users, 'n_samples': n_samples, 'n_dims': 50}

        # use the dict of data info defined above.
        data = recover_text_list(data_info['n_users'], data_info['n_samples'])
        tp = TextProcessor(data_info['n_users'], data_info['n_samples'], data_info['n_dims'])

        tfidf_corpus = tp.tf_idf_transform(data)
        # lsi_corpus = tp.lsi_transform(tfidf_corpus)
        lda_corpus = tp.lda_transform(tfidf_corpus)
        # w2v_corpus = tp.w2v_transform(data)

    # _test()



