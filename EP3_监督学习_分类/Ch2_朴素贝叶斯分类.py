from sklearn.datasets import fetch_20newsgroups  # 导入数据集
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.feature_extraction.text import TfidfVectorizer  # 文本特征抽取
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯算法
import pandas as pd
def nb_news():
    """
    用朴素贝叶斯算法对新闻数据集进行分类
    :return:
    """
    # 1）获取数据
    news = fetch_20newsgroups(data_home="../src")
    # 2）数据集划分
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.2, random_state=6)
    # 3）特征工程
    # tf-idf 文本特征抽取
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4）朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train) # 模型训练
    # 5）模型评估
    # 法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("预测值：\n", y_predict)
    print("真实值和预测值比对：\n", y_test == y_predict)
    # 法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None

if __name__ == '__main__':
    nb_news()