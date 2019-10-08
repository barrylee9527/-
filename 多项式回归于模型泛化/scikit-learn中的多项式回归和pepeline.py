import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
x = np.random.uniform(-3, 3, size=100)
x = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
print(x)
print(np.sort(x))
poly = PolynomialFeatures(degree=2)  # 多项式特征的数据集
poly.fit(x)
x2 = poly.transform(x)
print(x2.shape)
lin_reg2 = LinearRegression()
lin_reg2.fit(x2, y)
y_predict2 = lin_reg2.predict(x2)
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()
print(lin_reg2.coef_)
# 关于polynomialfeztures
x = np.arange(1, 11).reshape(-1, 2)
poly = PolynomialFeatures(degree=2)
poly.fit(x)
x2 = poly.transform(x)
# pipeline
x = np.random.uniform(-3, 3, size=100)
x = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

poly_reg = Pipeline([('poly', PolynomialFeatures(degree=2)), ('std_scaler', StandardScaler()), ('lin_reg', LinearRegression())])
poly_reg.fit(x, y)
y_predict = poly_reg.predict(x)
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()







