import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from 机器学习.逻辑回归.LogisticRegression import LogisticRegression
np.random.seed(666)
x = np.random.normal(0, 1, size=(200, 2))
y = np.array(x[:, 0]**2 + x[:, 1] < 1.5, dtype='int')
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.show()
# 使用逻辑回归
log_reg = LogisticRegression()
log_reg.fit(x, y)
score = log_reg.score(x, y)
print(score)
# plot_decision_boundary(model, axis) 写出来
def PolynomialLogisticRegression(degree):
    return Pipeline(
        [
            ('poly', PolynomialFeatures(degree)),
            ('std_scaler', StandardScaler()),
            ('log_reg', LinearRegression())
        ]
    )
poly_log_reg = PolynomialLogisticRegression(degree=20)
poly_log_reg.fit(x, y)
score = poly_log_reg.score(x, y)
print(score)
