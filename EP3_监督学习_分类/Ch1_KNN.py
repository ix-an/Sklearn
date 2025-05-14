from sklearn.datasets import load_iris                  # 导入数据集
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.preprocessing import StandardScaler        # 标准化
from sklearn.neighbors import KNeighborsClassifier      # KNN算法
import joblib

def knn_iris():
    """
    用KNN算法对鸢尾花数据集进行分类
    :return:
    """
    # 1)获取数据
    x,y = load_iris(return_X_y=True)

    # 2)划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)

    # 3）特征工程：标准化 --> KNN对量纲敏感
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)  # 训练集标准化
    x_test = transfer.transform(x_test)        # 测试集标准化

    # 4）KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)    # 训练模型

    # 5）模型评估：使用测试集
    # 方案1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接比对真实值和预测值：\n', y_test == y_predict)
    # 方案2：计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率为：\n', score)

    # 6) 保存模型
    joblib.dump(estimator, '../model/knn_iris.pkl')

    return None

if __name__ == '__main__':
    knn_iris()