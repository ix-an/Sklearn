"""
字典特征提取：字典列表向量化
"""
from sklearn.feature_extraction import DictVectorizer
data = [{'city':'扬州', 'temperature':27},
        {'city':'重庆', 'temperature':35},
        {'city':'成都', 'temperature':34},
        {'city':'上海', 'temperature':28}]

# 创建字典特征提取器
#transfer = DictVectorizer() # sparse默认为True，会返回稀疏矩阵
transfer = DictVectorizer(sparse=False) #返回ndarray

# 调用fit_transform()，应用特征提取器
data_new = transfer.fit_transform(data)
print("data_new:\n", data_new)
print("特征名字：\n", transfer.get_feature_names_out())