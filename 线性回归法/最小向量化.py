import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from 机器学习.线性回归法.playML.SimpleLinearRegression import SimpleLinearRegression1
from 机器学习.线性回归法.playML.SimpleLinearRegression import SimpleLinearRegression2
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
reg1 = SimpleLinearRegression1()
reg1.fit(x, y)
y_hat1 = reg1.predict(x)
plt.scatter(x, y)
plt.plot(x, y_hat1, color='r')
plt.axis([0, 6, 0, 6])
plt.show()
# 向量化实现SimpleLinearRegression
reg2 = SimpleLinearRegression2()
reg2.fit(x, y)
y_hat2 = reg2.predict(x)
plt.scatter(x, y)
plt.plot(x, y_hat2, color='r')
plt.axis([0, 6, 0, 6])
plt.show()
# 向量化的性能测试
m = 1000000
big_x = np.random.random(size=m)
big_y = big_x * 2 + 3 + np.random.normal(size=m)

reg1.fit(big_x, big_y)
reg2.fit(big_x, big_y)


