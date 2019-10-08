import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from 机器学习.逻辑回归.LogisticRegression import LogisticRegression
iris = datasets.load_iris()
x = iris.data
y = iris.target
x = x[y < 2, :2]
y = y[y < 2]
plt.scatter(x[y == 0, 0], x[y == 0, 1], color='r')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='b')
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
score = log_reg.score(x_test, y_test)
print(score)
print(log_reg.predict_proba(x_test))
print(y_test)
pre = log_reg.predict(x_test)
print(pre)
print(log_reg.coef_, log_reg.interception_)

