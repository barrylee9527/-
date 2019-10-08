import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
score1 = log_reg.score(x_test, y_test)
print(score1)
log_reg2 = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_reg2.fit(x_train, y_train)
score2 = log_reg2.score(x_test, y_test)
print(score2)
# 使用所有数据
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
score1 = log_reg.score(x_test, y_test)
print(score1)
log_reg2 = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_reg2.fit(x_train, y_train)
score2 = log_reg2.score(x_test, y_test)
print(score2)
# OvO and OvR
from sklearn.multiclass import OneVsRestClassifier
ovr = OneVsRestClassifier(log_reg)
ovr.fit(x_train, y_train)
score = ovr.score(x_test, y_test)
print(score)
from sklearn.multiclass import OneVsOneClassifier
ovr = OneVsOneClassifier(log_reg)
ovr.fit(x_train, y_train)
score = ovr.score(x_test, y_test)
print(score)
