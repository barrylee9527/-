from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
boston = datasets.load_boston()
x = boston.data
y = boston.target
x = x[y < 50.0]
y = y[y < 50.0]
# x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
"""knn_reg = KNeighborsRegressor()
knn_reg.fit(x_train, y_train)
score = knn_reg.score(x_test, y_test)
print(score)"""
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
knn_reg = KNeighborsRegressor()
grid_search = GridSearchCV(knn_reg, gram_grid, cv=3, verbose=1, iid='')
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_.score(x_test, y_test))