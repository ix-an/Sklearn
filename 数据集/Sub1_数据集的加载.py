from sklearn.datasets import load_iris

def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取鸢尾花数据集
    iris = load_iris()
    # print(iris)    # Bunch数据集，继承自字典

    # 鸢尾花的特征数据集(x,data,特征)
    data = iris.data
    print("数据集的特征值：\n",data[:5])
    print("数据集的特征值的数据类型：\n",type(data)) # numpy数组
    print("特征值的形状：\n",data.shape) # (150, 4)
    print("特征值的类型：\n", data.dtype) # float64
    print("特征值的名字:\n", iris.feature_names)
    """
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    sepal：花萼  petal：花瓣
    """

    print("----------------------------------------")

    # 鸢尾花的标签数据集(y,target,标签,labels,目标)
    target = iris.target
    print("数据集的标签值：\n",target) # 0,1,2
    print("数据集的标签值的数据类型：\n",type(target)) # numpy数组
    print("标签值的形状：\n",target.shape) # (150,)
    print("标签值的类型：\n", target.dtype) # int32
    print("标签值的名字:\n", iris.target_names)
    """
    ['setosa' 'versicolor' 'virginica']
    0:setosa(山鸢尾)  1:versicolor(变色鸢尾)  2:virginica(维基尼亚鸢尾)
    """

    print("----------------------------------------")
    print(iris.DESCR) # 数据集的描述
    print(iris.filename)  # 数据集的文件名
    return None
    

if __name__ == "__main__":
    datasets_demo()
