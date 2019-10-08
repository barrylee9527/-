import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from 机器学习.knn算法.playML.model_selection import train_test_split
from 机器学习.knn算法.playML.kNN import KNNClassifier
from 机器学习.knn算法.playML.metrics import accuracy_score
from sklearn import datasets
digits = datasets.load_digits()
digits.keys()
X = digits.data
y = digits.target
some_digit = X[666]
some_digit_image = some_digit.reshape(8, 8)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

my_knn_clf = KNNClassifier(k=3)
my_knn_clf.fit(X_train, y_train)
y_predict = my_knn_clf.predict(X_test)
d = sum(y_predict == y_test) / len(y_test)
# 封装我们自己的accuracy_score
score = accuracy_score(y_test, y_predict)
print(score)
