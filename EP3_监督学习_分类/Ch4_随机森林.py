import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def rf_train():
    # 加载数据
    data = pd.read_csv("../src/titanic/titanic.csv")
    x = data[["pclass", "age", "sex"]]
    # 填充缺失值
    x["age"].fillna(x["age"].mean(), inplace=True)
    x = x.to_dict(orient="records") # 转换为字典
    y = data["survived"].to_numpy() # 转换为numpy数组

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
    # 特征提取
    transfer = DictVectorizer(sparse=False)
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 模型训练：随机森林预估器
    estimator = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=22)
    estimator.fit(x_train, y_train)

    # 模型评估
    # 法1：直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print("预测值：\n", y_predict)
    print("真实值和预测值比对：\n", y_test == y_predict)
    # 法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 模型保存
    joblib.dump(transfer, "../model/titanic_rf_transfer.pkl")
    joblib.dump(scaler, "../model/titanic_rf_scaler.pkl")
    joblib.dump(estimator, "../model/titanic_rf_estimator.pkl")


def rf_predict():
    # 模拟数据
    data_new = [{"age": 30, "sex": "female", "pclass": 2}]
    # 加载模型
    transfer = joblib.load("../model/titanic_rf_transfer.pkl")
    scaler = joblib.load("../model/titanic_rf_scaler.pkl")
    estimator = joblib.load("../model/titanic_rf_estimator.pkl")
    # 数据预处理
    data_new = transfer.transform(data_new)
    data_new = scaler.transform(data_new)
    # 预测
    print("预测结果为：\n", estimator.predict(data_new))


if __name__ == "__main__":
    rf_train()
    rf_predict()

