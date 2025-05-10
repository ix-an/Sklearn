"""
中文的文本特征提取，需要分词（jieba分词）
"""
from sklearn.feature_extraction.text import CountVectorizer
import jieba

def cut_word(text):
    # jieba.cut()返回一个生成器，是一个可迭代对象
    # 而join()方法是将可迭代对象中的元素连接成字符串
    # 所以不需要先转换成list，再使用join()方法
    return " ".join(jieba.cut(text))

def  extract_text_demo():
    data = ["以前焦虑是自由的眩晕，现在焦虑是牛马的眩晕",
            "不要去焦虑9公里外和2小时后的事",
            "很遗憾你也懂焦虑的感受"]

    # 分词,使用推导式
    data_new = [cut_word(sent) for sent in data]
    print(data_new)

    # 创建文本特征提取器
    transfer = CountVectorizer()
    # 应用特征提取器
    data_final = transfer.fit_transform(data_new)
    print("data_final:\n", data_final.toarray())
    print("特征名：\n", transfer.get_feature_names_out())

    return None

if  __name__ == '__main__':
    extract_text_demo()
