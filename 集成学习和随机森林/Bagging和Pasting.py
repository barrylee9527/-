import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

x, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
# n_estimators指定子模型个数，max_sample指定每次训练传入的样本个数
bagging1_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True)
bagging1_clf.fit(X_train, y_train)
score1 = bagging1_clf.score(X_test, y_test)
bagging2_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=5000, max_samples=100, bootstrap=True)
bagging2_clf.fit(X_train, y_train)
score2 = bagging2_clf.score(X_test, y_test)
print(score1, score2)
