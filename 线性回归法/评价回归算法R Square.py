# MSE均方误差（mean squared error)
# RMSE均方根误差
# MAE(平均绝对误差)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
boston = datasets.load_boston()
x = boston.data[:, 5]
y = boston.target
x = x[y < 50.0]
y = y[y < 50.0]
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
x_mean = np.mean(x)
y_mean = np.mean(y)
num = (x_train - x_mean).dot(y_train - y_mean)
d = (x_train - x_mean).dot(x_train - x_mean)
a = num / d
b = y_mean - a * x_mean
print(a, b)
y_hat = a*x_train + b
x_predict = x_test
y_predict = x_test*a + b
plt.scatter(x_train, y_train)
plt.plot(x_train, y_hat, color='r')
plt.plot(x_predict, y_predict, color='y')
plt.show()
mse_test = np.sum((y_predict - y_test)**2)/len(y_test)
print(mse_test)
rmse_test = np.sqrt(mse_test)
print(rmse_test)
mae_test = np.sum(np.absolute(y_predict - y_test))/len(y_test)
print(mae_test)
mean_absolute_error(y_predict, y_test)
mean_squared_error(y_predict, y_test)
r_square = 1 - mean_squared_error(y_test, y_predict)/np.var(y_test)
print(r_square)