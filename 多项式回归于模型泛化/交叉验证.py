import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=k, p=p)
        knn_clf.fit(x_train, y_train)
        score = knn_clf.score(x_test, y_test)
        if score > best_score:
            best_score, best_p, best_k = score, p, k
print(best_k)
print(best_p)
print(best_score)
# 使用交叉验证
knn_clf = KNeighborsClassifier()
cross_val_score(knn_clf, x_train, y_train)
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=k, p=p)
        scores = cross_val_score(knn_clf, x_train, y_train)
        score = np.mean(scores)
        if score > best_score:
            best_score, best_p, best_k = score, p, k
print(best_k)
print(best_p)
print(best_score)
# 回顾网格搜索
from sklearn.model_selection import GridSearchCV
param_grid = [
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(2, 11)],
        'p': [i for i in range(1, 6)]
    }
]
grid_search = GridSearchCV(knn_clf, param_grid, verbose=1)
grid_search.fit(x_train, y_train)
print(grid_search.best_estimator_)
print(grid_search.best_score_, grid_search.best_params_)
best_knn_clf = grid_search.best_estimator_
best_knn_clf.score(x_test, y_test)
cross_val_score(knn_clf, x_train, y_train, cv=5)  # 分为5份
