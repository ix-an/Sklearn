from sklearn.linear_model import SGDRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def load_data():
    """
    加载加尼福利亚房价数据集，并研究数据集
    :return:
    """
    # 加载数据集
    path = "../src"
    data = fetch_california_housing(data_home=path)
    print("数据集形状:\n", data.data.shape)
    print("特征名：\n", data.feature_names)
    print("目标名：\n", data.target_names)
    return data


def sgd_demo(data):
    """
    SGD回归
    :param data:
    :return:
    """
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=6)
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 预估器
    estimator = SGDRegressor(eta0=0.01, learning_rate='invscaling', max_iter=10000)
    estimator.fit(x_train, y_train)
    # 得出模型
    print("权重系数为:\n", estimator.coef_)
    print("偏置项为:\n", estimator.intercept_)
    # 模型评估
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差为:\n", error)
    # 直接打分
    score = estimator.score(x_test, y_test)
    print("模型得分:\n", score)


if __name__ == '__main__':
    # 加载并研究数据集
    housing = load_data()
    # SGD回归
    sgd_demo(housing)
