"""
特征降维：Filter方式（过滤法）
    1. 低方差特征过滤
    2. 相关系数
"""
from lib2to3.btm_matcher import type_repr

from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import pandas as pd
def variance_demo():
    """
    低方差特征过滤
    :return:
    """
    # data可以是数组、list、DataFrame，也可以是读取的csv、excel文件
    data = pd.DataFrame([[0, 2, 0, 3],
                         [0, 1, 4, 3],
                         [0, 1, 1, 3]])

    # 创建转换器，设定阈值0.2
    selector = VarianceThreshold(threshold=0.2)
    # 调用fit_transform()方法，返回低方差特征过滤后的数据
    selected = selector.fit_transform(data)

    print(selected)
    print(selector.get_support())

    return None

def person_demo():
    """
    相关系数特征选择：皮尔逊相关系数
    :return:
    """
    # 读取数据
    data = pd.read_csv("../src/factor_returns.csv")
    data = data.iloc[:, 1: -2]
    print("data:\n",data)

    # 实例化一个转换器类
    selector = VarianceThreshold(threshold=10)
    # 调用fit_transform()
    selected = selector.fit_transform(data)
    print("data_new:\n",selected,selected.shape)

    # 计算某两个特征之间的相关系数
    r = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("相关系数：\n", r)
    return None



if __name__ == '__main__':
    #variance_demo()
    print("--------------------")
    person_demo()