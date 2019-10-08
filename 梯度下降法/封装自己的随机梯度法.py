import numpy as np
from 机器学习.线性回归法.LinearRegression import LinearRegression
m = 100
x = np.random.normal(size=m)
x = x.reshape(-1, 1)
y = 4. * x + 3. + np.random.normal(loc=0.0, scale=3.0, size=m)
lin_reg = LinearRegression()
lin_reg.fit_sgd(x, y, n_iters=2)
print(lin_reg.coef_)