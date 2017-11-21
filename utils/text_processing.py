# -*- coding: utf-8 -*-

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class TextProcessor():
    def __init__(self):
        # LDA.__init__(self, n_topics=n_topics, max_iter=max_iter, random_state=random_state)
        # self._super = super(TextProcessor, self)
        pass

    @staticmethod
    def tf_idf_transform(doc):
        if not len(doc):
            return [], []
        # print(doc)
        vectorizer = TfidfVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        try:
            # tfidf = transformer.fit_transform(vectorizer.fit_transform(doc))
            tf_idf_result = vectorizer.fit_transform(doc)
        except ValueError:
            return [], []

        word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        weight = tf_idf_result.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        #  打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        # for i in range(len(weight)):
        #         print("-------这里输出第", i, "类文本的词语tf-idf权重------")
        #         for j in range(len(word)):
        #             print(word[j], weight[i][j])
        # print(weight.shape)
        return word, weight


if __name__ == '__main__':
    result = TextProcessor()
