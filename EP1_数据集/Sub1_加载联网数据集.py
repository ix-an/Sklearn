from sklearn.datasets import fetch_20newsgroups
from sklearn import datasets

path = datasets.get_data_home()    # 查看数据集默认下载路径
print(path) # C:\Users\*****\scikit_learn_data

# "联网"加载新闻数据集
news = fetch_20newsgroups(data_home="../src", subset="all")
# print(news.data[0]) # 文本新闻信息
print(len(news.data)) # 18846
print(news.target) # 新闻的标签
print(news.target_names) # 新闻的分类
