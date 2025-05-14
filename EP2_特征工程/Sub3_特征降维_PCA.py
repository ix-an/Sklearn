from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# wine = load_wine()
# print(wine.data.shape)  # (178,13)
# print(wine.target)  # 三种酒类

def pca_demo(n_components=0.9):
    """
    主成分分析进行降维
    :return:
    """
    # 读取数据
    wine = load_wine()
    data = wine.data

    # 数据预处理：标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # PCA主成分分析 降维
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    print("降维后的数据：\n",data_pca,data_pca.shape)


if __name__ == '__main__':
    #pca_demo(0.95) # 保留95%的信息，降维后的维数：(178,10)
    pca_demo(5) # 降维到5个特征