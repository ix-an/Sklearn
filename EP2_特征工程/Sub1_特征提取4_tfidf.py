"""
用TF-IDF的方法进行文本特征提取
"""
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['I love programming',
          'Programming is fun']

vec = TfidfVectorizer()
X = vec.fit_transform(corpus)
print("TF-IDF矩阵：\n", X.toarray())
print("特征词：", vec.get_feature_names_out())
print("---------------------------------------")
"""
手动实现TF-IDF
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
def tfidf(data):
    # 计算词频
    count_vect = CountVectorizer()
    tf = count_vect.fit_transform(data).toarray()
    #print(tf)
    corpus_count = len(tf) # 文档总数
    term_count = np.sum(tf!=0, axis=0) # 每个词出现的文档数
    # 计算逆文档频率
    idf = np.log((corpus_count + 1)/(term_count + 1)) + 1
    # 计算TF-IDF
    tf_idf = tf * idf
    #print(tf_idf)
    # L2归一化
    tf_idf_norm = normalize(tf_idf, norm="l2")
    print(tf_idf_norm)

tfidf(corpus)


