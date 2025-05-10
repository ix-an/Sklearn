"""
特征预处理：标准化 StandardScaler
"""
from sklearn.preprocessing import StandardScaler

data = [[100, 10], [200, 20], [300, 30]]  # [人口百万, GDP万亿]
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
print(scaled)

"""
fit_transform() fit() transform() 的用法
"""
scaler.fit(data) # 训练：把data的均值和标准差保存下来
print("保存在转换器中的均值：",scaler.mean_)
print("保存在转换器中的标准差：",scaler.scale_)
scaler.transform(data) # 转换：把data进行标准化

data2 = [[10, 21],[17, 30], [20, 40]]
scaler.transform(data2) # 会使用data的均值和标准差进行转换

scaler.fit_transform(data2) # 会重新训练，使用data2的均值和标准差进行转换
data3 = [[20,20]]
scaler.transform(data3) # 使用data2的均值和标准差进行转换
"""
fit()和fit_transform() 应仅在训练集中使用
transform() 可以在任何数据集上使用，但是应用时的统计信息应该来自训练集
总结：先fit_transform(x_train)，再transform(x_test)
"""