"""
Kaggle实战：用户对物品类别的喜好细分
数据集：Instacart Market Basket Analysis
实战知识：特征工程

目标分析：找到用户和物品类别之间的关系
    1. 获取数据
    2. 合并表
    3. 找到user_id和aisles之间的关系
    4. PCA降维
"""
import pandas as pd
from sklearn.decomposition import PCA

# 获取数据
order_products = pd.read_csv("../src/order_products__prior.csv") #32434489× 4
products = pd.read_csv("../src/products.csv")  # (49688,4)
orders = pd.read_csv("../src/orders.csv")  # 3421083 rows × 7 columns
aisles = pd.read_csv("../src/aisles.csv") # (134,2)

# 合并表
# 合并aisles和products
tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])  # 49688 × 5 c
tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])  # 32434489 ,8
tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])  # 32434489 ,14

# 找到suer_id和aisle之间的关系
# 交叉表
table = pd.crosstab(tab3["user_id"], tab3["aisle"])  # 206209 rows × 134 columns
data = table[:100]  # 100 rows × 134 columns


# PCA降维
transfer = PCA(n_components=0.95)  # 保留95%的信息
data_new = transfer.fit_transform(data)  # (100, 42)，由134个特征降维到42个
