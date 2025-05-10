"""
加载本地自己的数据集
"""
import pandas as pd
# 读取csv文件
data = pd.read_csv("../src/ss.csv")
print(data) # 读取的是DataFrame格式
data = data.to_numpy() # 转换为numpy数组
print(data)
# 读取excel文件
excel = pd.read_excel("../src/excel_test.xlsx")
print(excel) # 也是DataFrame格式