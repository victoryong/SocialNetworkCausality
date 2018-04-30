# -*- coding: utf-8 -*-
"""
Create on Nov 22 Sat 2017

@author: Victor

A text processor whose aims at vectorizing text, includes tf-idf, lsi, lda, word2vec, etc.
"""

from io import StringIO
import os

import numpy as np
from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel
from gensim.models.word2vec import Word2Vec
from gensim import corpora, matutils

import utils.config_util as conf
from utils.log import get_console_logger

logger = get_console_logger(__name__)


class TextProcessor:
    def __init__(self):
        self.tfIdfModel = None
        self.lsiModel = None
        self.ldaModel = None
        self.w2vModel = None
        self.dictionary = None

        self.dictPath = conf.get_data_filename_via_template('model', model_type='tfidf', n_users=conf.N_USERS,
                                                            n_samples=conf.N_SAMPLES, model_filename='dict')
        self.tfIdfPath = conf.get_data_filename_via_template('model', model_type='tfidf', n_users=conf.N_USERS,
                                                             n_samples=conf.N_SAMPLES, model_filename='tfidf')
        self.lsiPath = conf.get_data_filename_via_template('model', model_type='lsi', n_users=conf.N_USERS,
                                                           n_samples=conf.N_SAMPLES, n_dims='{n_dims}',
                                                           model_filename='lsi_model')
        self.ldaPath = conf.get_data_filename_via_template('model', model_type='lda', n_users=conf.N_USERS,
                                                           n_samples=conf.N_SAMPLES, n_dims='{n_dims}',
                                                           model_filename='lda_model')
        self.w2vPath = conf.get_data_filename_via_template('model', model_type='w2v', n_users=conf.N_USERS,
                                                           n_samples=conf.N_SAMPLES, n_dims='{n_dims}',
                                                           model_filename='w2vmodel')
        self.w2vVecPath = conf.get_data_filename_via_template('model', model_type='w2v', n_users=conf.N_USERS,
                                                              n_samples=conf.N_SAMPLES, n_dims='{n_dims}',
                                                              model_filename='vec.txt')
        self.tfIdfCorpus = None

    def load_model(self, model_type, n_dims=conf.N_DIMS):
        model = None
        try:
            if model_type == 'tfidf':
                model = TfidfModel.load(self.tfIdfPath, mmap='r')
                self.tfIdfModel = model
            elif model_type == 'lsi':
                model = LsiModel.load(self.lsiPath.format(n_dims=n_dims), mmap='r')
                self.lsiModel = model
            elif model_type == 'lda':
                model = LdaModel.load(self.ldaPath.format(n_dims=n_dims), mmap='r')
                self.ldaModel = model
            elif model_type == 'w2v':
                model = Word2Vec.load(self.w2vPath.format(n_dims=n_dims), mmap='r')
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
        :param doc: Text list after segmentation.
        :return: tf-idf corpus
        """
        self.dictionary = corpora.Dictionary(doc)
        corpus = [self.dictionary.doc2bow(text) for text in doc]
        self.tfIdfModel = TfidfModel(corpus)

        conf.mk_dir(self.tfIdfPath)

        self.dictionary.save(self.dictPath)
        logger.info('Dictionary has been saved in %s.' % self.dictPath)

        self.tfIdfModel.save(self.tfIdfPath)
        logger.info('TF-IDF model has been saved in %s.' % self.tfIdfPath)

        self.tfIdfCorpus = self.tfIdfModel[corpus]
        tfidf_corpus_path = conf.get_data_filename_via_template('tfidf', n_users=conf.N_USERS,  postfix='mm',
                                                                n_samples=conf.N_SAMPLES)
        corpora.MmCorpus.serialize(tfidf_corpus_path, self.tfIdfCorpus)
        logger.info('TF-IDF corpus with a shape of %s has been saved in %s.' %
                    (np.array(self.tfIdfCorpus).shape, tfidf_corpus_path))

        return self.tfIdfCorpus

    def lsi_transform(self, corpus_tf_idf, n_topics=conf.N_DIMS):
        """
        Init a lsi model with a n_topics whose default is 500, then fit the model with corpus and transform it.
        :param corpus_tf_idf: tf-idf matrix
        :param n_topics: Number of topics.
        :return: lsi corpus
        """
        logger.info('Training lsi model with a n_dims of %d...' % n_topics)
        if self.dictionary is None and os.path.exists(self.dictPath):
            self.dictionary = corpora.Dictionary.load(self.dictPath)

        self.lsiModel = LsiModel(corpus=corpus_tf_idf, num_topics=n_topics, id2word=self.dictionary)
        # print self.lsiModel[corpus]

        lsi_path = self.lsiPath.format(n_dims=n_topics)
        conf.mk_dir(lsi_path)

        self.lsiModel.save(lsi_path)
        logger.info('Lsi model has been saved in %s.' % lsi_path)

        lsi_corpus = self.lsiModel[corpus_tf_idf]
        lsi_corpus_path = conf.get_data_filename_via_template('lsi', n_users=conf.N_USERS, n_samples=conf.N_SAMPLES,
                                                              n_dims=n_topics, postfix='mm')
        conf.mk_dir(lsi_corpus_path)
        corpora.MmCorpus.serialize(lsi_corpus_path, lsi_corpus)
        logger.info(
            'Lsi corpus with a shape of %s has been saved in %s.' % (np.array(lsi_corpus).shape, lsi_corpus_path))

        return lsi_corpus

    def lda_transform(self, corpus_tf_idf, n_topics=conf.N_DIMS, train_separated=False, is_update=False):
        """
        Init a lda model with a n_topics whose default is 500, then fit it with corpus_tf_idf and transform it.
        :param corpus_tf_idf: Corpus which has been transformed into tf-idf matrix.
        :param n_topics: Number of topics, default is 500.
        :param train_separated: The model is going to be train with all corpus one time or some of them separately one time.
        :param is_update: Whether the training to be perform is to construct a new model or update one existed.
        :return: lda corpus.
        """
        logger.info('Training lda model with a n_dims of %d...' % n_topics)
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

        self.ldaModel = LdaModel(corpus=corpus_tf_idf, num_topics=n_topics, id2word=self.dictionary)
        lda_path = self.ldaPath.format(n_dims=n_topics)
        conf.mk_dir(lda_path)
        self.ldaModel.save(lda_path)
        logger.info('lda model has been saved in %s' % lda_path)

        lda_corpus = self.ldaModel[corpus_tf_idf]
        lda_corpus_path = conf.get_data_filename_via_template('lda', n_users=conf.N_USERS, n_samples=conf.N_SAMPLES,
                                                              n_dims=n_topics, postfix='mm')
        conf.mk_dir(lda_corpus_path)
        corpora.MmCorpus.serialize(lda_corpus_path, lda_corpus)
        logger.info(
            'Lda corpus with a shape of %s has been saved in %s.' % (np.array(lda_corpus).shape, lda_corpus_path))

        return lda_corpus

    def w2v_transform(self, sentences, n_dims=conf.N_DIMS):
        """
        Perform word2vec on texts and obtain a w2v model.
        :param sentences: Sentences that each one of it contains a list of words of a text.
        :param n_dims: Size of w2v.
        :return: W2v model.
        """
        logger.info('Training w2v model with a n_dims of %d...' % n_dims)
        # file = open(infile_path, 'r', encoding='utf-8') if infile_path.find('\n') < 0 else StringIO(infile_path)
        # sentences = []
        # for sen in file.readlines():
        #     sentences.append(sen.strip().split(' '))
        # print(sentences)
        self.w2vModel = Word2Vec(sentences, size=n_dims, min_count=0)

        w2v_path = self.w2vPath.format(n_dims=n_dims)
        w2v_vec_path = self.w2vVecPath.format(n_dims=n_dims)
        conf.mk_dir(w2v_path)
        self.w2vModel.save(w2v_path)
        self.w2vModel.wv.save_word2vec_format(w2v_vec_path, binary=False)
        # print(model['['])

        # Construct w2v corpus
        w2v_corpus = []
        for sen in sentences:
            vec = [0] * n_dims
            if len(sen) > 0:
                for word in sen:
                    vec = list(map(lambda m, n: m + n, vec, self.w2vModel[word]))
                    # vec += self.w2vModel[word]
            w2v_corpus.append(vec)

        return w2v_corpus

    def load_corpus(self, model_type, n_dims=conf.N_DIMS, dense=False):
        corpus = None
        if model_type == 'tfidf':
            corpus = corpora.MmCorpus(
                conf.get_data_filename_via_template(
                    'tfidf', n_users=conf.N_USERS, postfix='mm', n_samples=conf.N_SAMPLES))
            self.tfIdfCorpus = corpus
        elif model_type in ['lsi', 'lda']:
            corpus = corpora.MmCorpus(conf.get_data_filename_via_template(
                model_type, n_users=conf.N_USERS, n_samples=conf.N_SAMPLES, n_dims=n_dims, postfix='mm'))
        elif model_type == 'w2v':
            pass

        logger.info('%s corpus with a shape of %s has been loaded. ' % (model_type, np.array(corpus).shape))
        if dense and model_type in ['tfidf', 'lsi', 'lda']:
            if self.dictionary is None and os.path.exists(self.dictPath):
                self.dictionary = corpora.Dictionary.load(self.dictPath)
            corpus = matutils.corpus2dense(corpus, self.dictionary.num_pos, conf.N_SAMPLES)
        else:
            corpus = np.array(corpus)
        return corpus

    @staticmethod
    def corpus2dense(corpus, n_terms, n_docs=conf.N_SAMPLES):
        return matutils.corpus2dense(corpus, n_terms, n_docs)

if __name__ == '__main__':
    tp = TextProcessor()
    # corpus = ['你好 中国', '打印 每类 文本 for', 'for', '遍历 所有 文本 第二个 for 便利 某一类 文本 下 的 词语 权重']
    # print(tp.w2v_transform('\n'.join(corpus))['你好'])
    # corpus = ['你好 中国'.split(), '打印 每类 文本 for'.split(), 'for'.split(), '遍历 所有 文本 第二个 for 便利 某一类 文本 下 的 词语 权重'.split()]
    # ctf = tp.tf_idf_transform(corpus)
    # clsi = tp.lda_transform(ctf)
    #
    # for i in clsi:
    #     print(i)
    # aa = tp.ldaModel.print_topics(num_topics=500, num_words=50)
    # for i in aa:
    #     print(i)

    # path = conf.get_data_filename_via_template(
    #         'model',
    #         model_type='lsi',
    #         n_users=conf.N_USERS,
    #         n_samples=conf.N_SAMPLES,
    #         n_dims=conf.N_DIMS)
    # tp.load_model('lsi')
    tp.load_corpus('tfidf')
    import numpy as np
    print(np.array(tp.tfIdfCorpus).shape)


