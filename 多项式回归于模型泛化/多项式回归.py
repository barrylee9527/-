import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.random.uniform(-3, 3, size=100)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
x = x.reshape(-1, 1)
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_predict = lin_reg.predict(x)
plt.scatter(x, y)
plt.plot(x, y_predict, color='r')
plt.show()
# 解决方案，添加一个特征
x2 = np.hstack([x, x**2])
lin_reg2 = LinearRegression()
lin_reg2.fit(x2, y)
y_predict2 = lin_reg2.predict(x2)
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()
