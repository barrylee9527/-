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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
score = log_reg.score(x_test, y_test)
def x2(x1):
    return (-log_reg.coef_[0] * x1 - log_reg.interception_) / log_reg.coef_[1]
x1_plot = np.linspace(4, 8, 1000)
x2_plot = x2(x1_plot)
plt.scatter(x[y == 0, 0], x[y == 0, 1], color='r')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='b')
plt.plot(x1_plot, x2_plot)
plt.show()
plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color='r')
plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color='b')
plt.plot(x1_plot, x2_plot)
plt.show()
# KNN的决策边界
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)
