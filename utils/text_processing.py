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



class TextProcessor:
    def __init__(self, max_features=None, min_df=1, max_df=1.0):
        # LDA.__init__(self, n_topics=n_topics, max_iter=max_iter, random_state=random_state)
        # self._super = super(TextProcessor, self)
        self.vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频

    def tf_idf_transform(self, doc):
        if not len(doc):
            return [], []

        # print(doc)
        # self.vectorizer = TfidfVectorizer(max_features=None, min_df=1, max_df=1.0)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        try:
            # tfidf = transformer.fit_transform(self.vectorizer.fit_transform(doc))
            tf_idf_result = self.vectorizer.fit_transform(doc)
        except ValueError:
            return [], []

        word = self.vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        weight = tf_idf_result.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        # for i in range(len(weight)):
        #         print("-------这里输出第", i, "类文本的词语tf-idf权重------")
        #         for j in range(len(word)):
        #             print(word[j], weight[i][j])
        # print(weight.shape)
        return word, weight


if __name__ == '__main__':
    result = TextProcessor(max_features=11, min_df=1, max_df=1.0)
    w, wei = result.tf_idf_transform(['打印 每类 文本 for', 'for', '遍历 所有 文本，第二个 for 便利 某一类 文本 下 的 词语 权重'])
    import numpy as np
    print(np.sum(wei))
