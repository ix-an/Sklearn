from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 获取数据
x, y = load_iris(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=6)

# 标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 创建 knn模型
knn_model = KNeighborsClassifier(n_neighbors=7)
# 添加网格搜索和交叉验证
param_dict = {"n_neighbors":[3,5,7,9,11]}
model = GridSearchCV(knn_model, param_grid=param_dict, cv=4)

# 模型训练
model.fit(x_train,y_train)

print("最佳参数:",model.best_params_)
print("最佳结果:",model.best_score_)
print("最佳模型:",model.best_estimator_)
print("交叉验证结果信息:",model.cv_results_)
print("最佳K值下标",model.best_index_)

# 模型评估
y_pred = model.predict(x_test)
print("预测值为：\n",y_pred)
print("比对真实值和预测值：\n",y_test==y_pred)
score = model.score(x_test,y_test)
print("准确率为：\n",score)

"""
为什么准确率和最佳结果不一样？-->使用的数据不一样
最佳结果是用交叉验证的结果计算出来的最佳的一次结果
准确率是用测试集计算出来的验证结果
"""