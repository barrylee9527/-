import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from 机器学习.线性回归法.LinearRegression import LinearRegression
boston = datasets.load_boston()
x = boston.data
y = boston.target
x = x[y < 50.0]
y = y[y < 50.0]
# x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)


def J(theta, x_b, y):
    try:
        return np.sum((y - x_b.dot(theta))**2) / len(x_b)
    except:
        return float('inf')


def dJ(theta, x_b, y):
    return x_b.T.dot(x_b.dot(theta) - y) * 2 / len(x)


def gradint_descent(x_b, y, initial_theat, eta, n_iters=1e4, epsilon=1e-8):
    theta = initial_theat
    i_iter = 0
    while i_iter < n_iters:
        gradient = dJ(theta, x_b, y)
        last_theat = theta
        theta = theta - eta * gradient
        if abs(J(theta, x_b, y) - J(last_theat, x_b, y)) < epsilon:
            break
        i_iter += 1
    return theta
# lin_reg1 = LinearRegression()
# lin_reg1.fit_normal(x_train, y_train)
# lin_reg1.fit_gd(x_train, y_train, eta=0.000001)
# score1 = lin_reg.score(x_test, y_test)
# print(lin_reg.coef_)
# print(lin_reg.interception_)
# print(score1)
# lin_reg2.fit_gd(x_train, y_train, eta=0.000001, n_iters=1e6)
# score2 = lin_reg.score(x_test, y_test)
# print(score)
# 结果还是不理想，而且耗时，在训练前进行数据归一化
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(x_train)
x_train_standard = standardScaler.transform(x_train)
x_test_standard = standardScaler.transform(x_test)
lin_reg3 = LinearRegression()
lin_reg3.fit_gd(x_train_standard, y_train)
score3 = lin_reg3.score(x_test_standard, y_test)
print(score3)
