import matplotlib.pyplot as plt
import numpy as np

def model(x,w):
    return np.dot(x,w)
def loss(w):
    return 10 * (w**2) + 15.9 * w + 6.5
def gradient(w):
    return 20 * w + 15.9


# 梯度下降
w = np.random.randint(-10,10)    # 随机初始化 w
print("初始化的w：", w)
alpha = 0.1    # 学习率
i = 0
arr_w = np.array([])
while True:
    i += 1
    w = w - alpha * gradient(w)
    if i % 5 == 0: arr_w = np.append(arr_w, w)
    if i % 100 == 0:
        print(f"第{i}次迭代：loss={loss(w)}")
    if abs(loss(w)) < 0.05:
        print(f"第{i}次迭代：loss={loss(w)}")
        print(f"w={w}")
        break

# 绘制损失函数
a = np.linspace(-10,10,1000)
plt.plot(a,loss(a))
plt.scatter(arr_w,loss(arr_w),c='r')
plt.show()