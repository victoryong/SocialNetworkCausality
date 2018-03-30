# -*- coding: utf-8 -*-
"""
Create on Nov 22 Sat 2017

@author: Victor Y, Xie

TextProcessor to execute TF-IDF and LDA on texts.
"""

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora

import utils.config_util as conf
from utils.log import get_console_logger

logger = get_console_logger(__name__)

class TextProcessor:
    def __init__(self, max_features=None, min_df=1, max_df=1.0, stop_words=None):
        # LDA.__init__(self, n_topics=n_topics, max_iter=max_iter, random_state=random_state)
        # self._super = super(TextProcessor, self)

        # self.vectorizer = TfidfVectorizer(max_features=max_features,
        #                                   min_df=min_df,
        #                                   # stop_words=stop_words,
        #                                   max_df=max_df,
        #                                   )  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        self.tfIdfModel = None
        self.lsiModel = None
        self.dictionary = None

    # def tf_idf_transform(self, doc):
    #     if not len(doc):
    #         return [], []
    #
    #     # print(doc)
    #     # self.vectorizer = TfidfVectorizer(max_features=None, min_df=1, max_df=1.0)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    #     # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    #     try:
    #         # tfidf = transformer.fit_transform(self.vectorizer.fit_transform(doc))
    #         tf_idf_result = self.vectorizer.fit_transform(doc)
    #     except ValueError:
    #         return [], []
    #
    #     word = self.vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    #     weight = tf_idf_result.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    #     # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     # for i in range(len(weight)):
    #     #         print("-------这里输出第", i, "类文本的词语tf-idf权重------")
    #     #         for j in range(len(word)):
    #     #             print(word[j], weight[i][j])
    #     # print(weight.shape)
    #     return word, weight

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
