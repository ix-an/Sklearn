from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def train():
    """
    岭回归对加尼福利亚房价数据进行预测
    :return:
    """
    # 1.获取数据
    x, y = fetch_california_housing(data_home="../src", return_X_y=True)

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 3. 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. 岭回归预估器
    ridge = Ridge(alpha=1.0, solver="auto")
    ridge.fit(x_train, y_train)

    # 5. 得出模型
    print("权重系数为:\n", ridge.coef_)
    print("偏置项为:\n", ridge.intercept_)

    # 6. 模型评估
    y_predict = ridge.predict(x_test)
    print("预测房价为:\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差为:\n", error)

    return None

if __name__ == "__main__":
    train()