# -*- coding: utf-8 -*-
"""
Create on Nov 22 Sat 2017

@author: Victor

TextProcessor to execute TF-IDF and LDA on texts.
"""

from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora

import utils.config_util as conf
from utils.log import get_console_logger

logger = get_console_logger(__name__)


class TextProcessor:
    def __init__(self, max_features=None, min_df=1, max_df=1.0, stop_words=None):
        self.tfIdfModel = None
        self.lsiModel = None
        self.dictionary = None

    def load_model(self, model_type):
        fname = conf.get_data_filename_via_template('model', model_type=model_type)
        if model_type == 'tfidf':
            self.tfIdfModel = TfidfModel.load(fname, mmap='r')
        elif model_type == 'lsi':
            self.lsiModel = LsiModel.load(fname, mmap='r')
        else:
            logger.error('Model type error. Unexpected %s' % model_type)

    def tf_idf_transform(self, doc):
        """
        Perform tf-idf on doc.
        :param doc: Test list after segmentation.
        :return: tf-idf of doc.
        """
        self.dictionary = corpora.Dictionary(doc)
        corpus = [self.dictionary.doc2bow(text) for text in doc]
        self.tfIdfModel = TfidfModel(corpus)
        self.dictionary.save(conf.get_data_filename_via_template('model', model_type='dict'))
        self.tfIdfModel.save(conf.get_data_filename_via_template('model', model_type='tfidf'))
        # for i in self.tfIdfModel[corpus]:
        #     print i
        return self.tfIdfModel[corpus]

    def lsi_transform(self, corpus_tf_idf, n_topics=100):
        """
        Init a lsi model, fit the model with corpus and transform it.
        :param corpus: tf-idf matrix
        :param n_topics: Number of topics.
        :return: lsi result
        """
        self.lsiModel = LsiModel(corpus=corpus_tf_idf, num_topics=n_topics, id2word=self.dictionary)
        # print self.lsiModel[corpus]
        self.lsiModel.save(conf.get_data_filename_via_template('model', model_type='lsi'))
        return self.lsiModel[corpus_tf_idf]

    def lda_transform(self, corpus):
        """
        Init a lda model, fit the model with corpus and transform it.
        :param corpus: tf-tdf matrix
        :return: lda result
        """
        pass


if __name__ == '__main__':
    tp = TextProcessor()
    corpus = ['你好 中国'.split(), '打印 每类 文本 for'.split(), 'for'.split(), '遍历 所有 文本，第二个 for 便利 某一类 文本 下 的 词语 权重'.split()]
    ctf = tp.tf_idf_transform(corpus)
    clsi = tp.lsi_transform(ctf)
    for i in clsi:
        print(i)
