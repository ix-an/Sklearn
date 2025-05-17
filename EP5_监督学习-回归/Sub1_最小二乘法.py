from sklearn.linear_model import LinearRegression
import numpy as np

# 数据：包含x和y，y是心理健康指数，x是各种因素
data=np.array([[0,14,8,0,5,-2,9,-3,399],
               [-4,10,6,4,-14,-2,-14,8,-144],
               [-1,-6,5,-12,3,-3,2,-2,30],
               [5,-2,3,10,5,11,4,-8,126],
               [-15,-15,-8,-15,7,-4,-12,2,-395],
               [11,-10,-2,4,3,-9,-6,7,-87],
               [-14,0,4,-3,5,10,13,7,422],
               [-3,-7,-2,-8,0,-6,-5,-9,-309]])
# 分离出x和y
x,y = data[:,:-1], data[:,-1]

# 创建线性回归模型，进行训练
model = LinearRegression() # 默认使用最小二乘法
model.fit(x,y)

# 输出训练结果：w和b
w,b = model.coef_, model.intercept_
print("权重w：\n", w)
print("截距b：\n", b)
# 预测新数据
X = np.array([[0,14,8,0,5,-2,9,-3]]) # 399
#y_pred = model.predict(X)
y_pred = X @ w.T + b
print("预测结果：\n", y_pred) # [399.]
