import numpy as np
import matplotlib.pyplot as plt
from 机器学习.线性回归法.LinearRegression import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
boston = datasets.load_boston()
x = boston.data
y = boston.target
x = x[y < 50.0]
y = y[y < 50.0]
# x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
reg = LinearRegression()
reg.fit_normal(x_train, y_train)
print(reg.coef_)
print(reg.interception_)
score = reg.score(x_test, y_test)
print(score)
