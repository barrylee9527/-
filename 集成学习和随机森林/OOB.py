import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()


bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                                max_samples=100, bootstrap=True, oob_score=True)
bagging_clf.fit(X, y)
score = bagging_clf.oob_score_
print(score)
# n_jobs 使用计算机的核
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                                bootstrap=True, oob_score=True)
bagging_clf.fit(X, y)
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                                bootstrap=True, oob_score=True, n_jobs=-1)
bagging_clf.fit(X, y)
# bootstrap_features
random_subspaces_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=500,
                                         bootstrap=True, oob_score=True, max_features=1, bootstrap_features=True)
random_subspaces_clf.fit(X, y)
score1 = random_subspaces_clf.oob_score_
random_patches_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                                       bootstrap=True, oob_score=True, max_features=1, bootstrap_features=True)
random_patches_clf.fit(X, y)
score2 = random_patches_clf.oob_score_
