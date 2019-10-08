from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split   # 切分
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.metrics import accuracy_score   # 准确度
from sklearn import datasets
"""knn = KNeighborsClassifier(3)
x_train = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
y_train = ['A', 'A', 'B', 'B']
knn.fit(x_train, y_train)
x = np.array([0.1, 0.0])
x_t = x.reshape(1, -1)   # -1表示自动决定第二个维度
pre = knn.predict(x_t)
print(pre[0])"""
digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_text = train_test_split(x, y, test_size=0.2, random_state=666)
"""knn = KNeighborsClassifier(n_neighbors=4, weights='uniform')
knn.fit(x_train, y_train)
acc = knn.score(x_test, y_text)
print(acc)"""
gram_grid = [{
    'weights': ['uniform'],
    'n_neighbors': [i for i in range(1, 11)]
},
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, gram_grid, n_jobs=2, verbose=1)
grid_search.fit(x_train, y_train)
# print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)
