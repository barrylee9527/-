import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
np.random.seed(666)
x = np.random.uniform(-3, 3, size=100)
x = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_predict = lin_reg.predict(x)
mse = mean_squared_error(y, y_predict)
print(mse)
def PolynomialRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('lin_reg', LinearRegression())
    ])
poly_reg = PolynomialRegression(degree=2)
poly_reg.fit(x, y)
y2_predict = poly_reg.predict(x)
mse = mean_squared_error(y, y2_predict)
print(mse)
plt.scatter(x, y)
plt.plot(np.sort(x), y2_predict[np.argsort(x)], color='r')
plt.show()
poly_reg100 = PolynomialRegression(degree=100)
x_plot = np.linspace(-3, 3, 100).reshape(100, 1)
y_plot = poly_reg100.predict(x_plot)
plt.scatter(x, y)
plt.plot(x_plot[:, 0], y_plot, color='r')
plt.axis([-3, 3, -1, 10])
plt.show()





