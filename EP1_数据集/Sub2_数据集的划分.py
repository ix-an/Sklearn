"""
数据集的划分：
    训练集：用于训练模型的数据集 - 80% 70%
    测试集：用于评估模型性能的数据集 - 20% 30%
    验证集、交叉验证（暂略）
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def datasets_demo():
    """
    数据集的划分
    :return:
    """
    # 加载数据集
    iris = load_iris()
    x = iris.data
    y = iris.target
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
    print("训练集的形状：\n",x_train.shape) # (120, 4)
    print("训练集的标签的形状：\n",y_train.shape) # (120,)
    print("测试集的形状：\n",x_test.shape) # (30, 4)
    print("测试集的标签的形状：\n",y_test.shape)  # (30,)
    return None

if __name__ == '__main__':
    datasets_demo()
