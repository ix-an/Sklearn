from sklearn.preprocessing import MinMaxScaler

data = [[90, 170], [60, 160], [30, 150]]  # [成绩, 身高]
# 创建转换器（可指定范围）
scaler = MinMaxScaler(feature_range=(0,1))
# 应用转换器
scaled = scaler.fit_transform(data)
print(scaled)


from sklearn.preprocessing import normalize

data = [[3, 4], [1, 1]]  # 两个学生的两科排名
normalized = normalize(data,norm='l2',axis=1)
print(normalized)