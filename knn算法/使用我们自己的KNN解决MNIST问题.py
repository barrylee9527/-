import numpy as np
from sklearn.datasets import fetch_mldata
from 机器学习.knn算法.playML.kNN import KNNClassifier


mnist = fetch_mldata('MNIST original')


X, y = mnist['data'], mnist['target']
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)
knn_clf = KNNClassifier(k=5)

knn_clf.fit(X_train, y_train)
# 由于MNIST数据集巨大，直接进行计算，时间将异常高。
score = knn_clf.score(X_test, y_test)

