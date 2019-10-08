import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
boston = datasets.load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
dt_reg = DecisionTreeRegressor()
dt_reg.fit(x_train, y_train)
score1 = dt_reg.score(x_test, y_test)
score2 = dt_reg.score(x_train, y_train)
print(score1, score2)