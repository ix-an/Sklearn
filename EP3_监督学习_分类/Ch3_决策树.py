"""
学习案例：用决策树实现泰坦尼克号乘客生存预测
流程分析：特征值、目标值
1）获取数据
2）数据处理：缺失值处理，特征值->字典类型，
3）准备好特征值、目标值
4）划分数据集
5）特征工程：字典特征处理
6）决策树预估器流程
7）模型评估
"""
import pandas as pd  # 数据处理
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.feature_extraction import DictVectorizer # 字典特征抽取
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 决策树预估器，决策树可视化


# 1）获取数据
path = "../src/titanic/titanic.csv"
titanic = pd.read_csv(path)

# 筛选特征值和目标值
x = titanic[["pclass", "age", "sex"]]
y = titanic["survived"]

# 2）数据处理
# 缺失值处理
x["age"].fillna(x["age"].mean(), inplace=True)
# 转换为字典
x = x.to_dict(orient="records")

# 3）数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=22)

# 4）字典特征抽取
transfer = DictVectorizer(sparse=False)
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 5）决策树预估器流程
estimator = DecisionTreeClassifier(criterion="entropy")
estimator.fit(x_train, y_train)

# 6）模型评估
# 法1：直接对比真实值和预测值
y_predict = estimator.predict(x_test)
print("预测值：\n", y_predict)
print("真实值和预测值比对：\n", y_test == y_predict)
# 法2：计算准确率
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)

# 7）决策树可视化
export_graphviz(estimator,
                out_file="../model/titanic_tree.dot",
                feature_names=transfer.get_feature_names_out())












