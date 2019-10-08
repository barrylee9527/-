import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(42)
x = np.random.uniform(-3, 3, size=100)
# x = x.reshape(-1, 1)
y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)
# plt.scatter(x, y)
# plt.show()
x = x.reshape(-1, 1)
def PolynomialRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('lin_reg', LinearRegression())
    ])
np.random.seed(666)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
poly10_reg = PolynomialRegression(degree=20)
poly10_reg.fit(x_train, y_train)
y10_predict = poly10_reg.predict(x_test)
score = mean_squared_error(y_test, y10_predict)
x_plot = np.linspace(-3, 3, 100).reshape(100, 1)
y_plot = poly10_reg.predict(x_plot)
plt.scatter(x, y)
plt.plot(x_plot[:, 0], y_plot, color='r')
plt.axis([-3, 3, -1, 10])
plt.show()
def plot_model(model):
    x_plot = np.linspace(-3, 3, 100).reshape(100, 1)
    y_plot = model.predict(x_plot)
    plt.scatter(x, y)
    plt.plot(x_plot[:, 0], y_plot, color='r')
    plt.axis([-3, 3, -1, 10])
    plt.show()
# 使用岭回归
from sklearn.linear_model import Ridge
def RidgeRegression(degree, alpha):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('lin_reg', Ridge(alpha=alpha))
    ])
ridge1_reg = RidgeRegression(20, 0.0001)
ridge1_reg.fit(x_train, y_train)
y1_predict = ridge1_reg.predict(x_test)
score1 = mean_squared_error(y_test, y1_predict)
print(score1)
plot_model(ridge1_reg)
ridge2_reg = RidgeRegression(20, 20)
ridge2_reg.fit(x_train, y_train)
y2_predict = ridge2_reg.predict(x_test)
score2 = mean_squared_error(y_test, y2_predict)
print(score2)
plot_model(ridge2_reg)