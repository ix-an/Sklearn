"""
文本特征提取：词频特征提取
"""
from sklearn.feature_extraction.text import CountVectorizer

data = ['life is short,i like like python',
            'life is too long,i dislike python']
# 实例化转换器类
transfer = CountVectorizer(stop_words=["dislike"])
# 调用fit_transform()，应用特征提取器
data_new = transfer.fit_transform(data)
print("data_new:\n", data_new.toarray())    # toarray()将稀疏矩阵转为数组
print("特征名字：\n", transfer.get_feature_names_out())

