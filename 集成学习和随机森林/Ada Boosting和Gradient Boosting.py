# Boosting
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(X[y == 0, 0], X[y ==0, ])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
# AdaBoosting
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2), n_estimators=500)
ada_clf.fit(X_train, y_train)
score1 = ada_clf.score(X_test, y_test)
print(score1)
# Gradient Boosting
gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=30)
gb_clf.fit(X_train, y_train)
score2 = gb_clf.score(X_test, y_test)
print(score2)
# Boosting 解决回归问题
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

