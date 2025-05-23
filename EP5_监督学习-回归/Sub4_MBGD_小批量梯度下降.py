"""
小批量梯度下降：没有API，需要自己实现，使用SGDRegressor
partial_fit() 函数，可以继续训练，而不是每次都从头开始训练
"""
from sklearn.datasets import fetch_california_housing  # 数据集
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.linear_model import SGDRegressor  # 小批量梯度下降
from sklearn.metrics import mean_squared_error  # 均方误差
import numpy as np
import os  # 文件操作
import joblib  # 模型保存和加载
import math  # 数学运算
import time  # 时间

# 如果模型在本地有，就加载出来继续训练（保存的模型 -> 曾经训练修改过多次的 w和 b）
# 如果没有模型，就创建一个模型，第一次训练
# os.path.join() 会自动把路径拼接起来，并且会自动把路径分隔符给加上
model_path = os.path.join(os.path.dirname(__file__), "model", "mbgd_regressor_model.pkl")
transfer_path = os.path.join(os.path.dirname(__file__), "model", "mbgd_standardscaler_model.pkl")


def train():
    """
    训练模型：用BGDRegressor模拟MBGD，对加尼福利亚房价数据集进行训练
    :return:
    """
    # 0.加载模型
    model = None
    transfer = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        transfer = joblib.load(transfer_path)
    else:
        model = SGDRegressor(max_iter=100, learning_rate='constant', eta0=0.001)
        transfer = StandardScaler()

    # 1.获取数据
    x, y = fetch_california_housing(data_home="../src", return_X_y=True)  # (20640, 8)

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)

    # 3.标准化
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.训练模型
    epochs = 10  # 迭代次数(训练次数)
    batch_size = 32  # 批量大小 (每次训练的样本数量)
    # 外层循环控制迭代次数
    for epoch in range(epochs):
        indices = np.arange(len(x_train))    # 训练集数据的索引
        np.random.shuffle(indices)    # 打乱索引
        start_time = time.time()

        # 内层循环的次数取决于样本数和批量大小
        for batch in range(math.ceil(len(x_train) / batch_size)):
            start = batch * batch_size
            end = min(start + batch_size, len(x_train))    # 防止越界
            index = indices[start:end]    # 当前批次的索引
            x_batch = x_train[index]
            y_batch = y_train[index]
            """
            本来应该用这32条数据计算损失函数，然后计算梯度，再做梯度下降
            但是因为没有BGD的API，所以我们使用SGD的API来模拟
            后期深度学习时，会做出真正的BGD
            所以别在乎准确率和MSE了，关注整个代码的结构
            """
            model.partial_fit(x_batch, y_batch)    # 继续训练

        # 每一次迭代后，监测损失函数的均方误差
        y_predict = model.predict(x_test)
        error = mean_squared_error(y_test, y_predict)
        print(f"训练轮次：{epoch}/{epochs}，---> MSE = {error}", end=" ")
        print(f"模型得分：{model.score(x_test, y_test)}", end=" ")
        print(f"本轮训练时长：{time.time() - start_time:.2f}")

        # 5. 保存模型
        joblib.dump(model, model_path)
        joblib.dump(transfer, transfer_path)
        """
        这里也应该设置：如果模型更好，才保存模型（best_error）
        pytorch框架中，有torch.save，可以保存模型的参数，而不是整个模型
        """
    print("训练结束")


def detect():
    """
    模型预测：用训练好的模型，去预测新数据
    :return:
    """
    model = joblib.load(model_path)
    transfer = joblib.load(transfer_path)
    x_true = np.array([[3.3222, 22.0, 8.3252, 0.5734, 322.0, 3.7073, -122.2, 4.5737]])
    x_true = transfer.transform(x_true)
    y_predict = model.predict(x_true)
    print("预测结果：", y_predict)


if __name__ == '__main__':
    train()
    detect()
