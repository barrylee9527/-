from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.linear_model import SGDRegressor
from 机器学习.线性回归法.LinearRegression import LinearRegression
boston = datasets.load_boston()
x = boston.data
y = boston.target
x = x[y < 50.0]
y = y[y < 50.0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
standardScaler = StandardScaler()
standardScaler.fit(x_train)
x_train_standard = standardScaler.transform(x_train)
x_test_standard = standardScaler.transform(x_test)
"""
start = time.time()
lin_reg.fit_sgd(x_train_standard, y_train, n_iters=20)
score = lin_reg.score(x_test_standard, y_test)
end = time.time()
print('消耗时间%fs' % (end - start))
print(score)"""
sgd_reg = SGDRegressor(max_iter=100, n_iter=100)  # 新版本中n_iter被弃用(has been deprecated)
sgd_reg.fit(x_train_standard, y_train)
score = sgd_reg.score(x_test_standard, y_test)
print(score)
