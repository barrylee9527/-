import numpy as np
import time
from 机器学习.线性回归法.LinearRegression import LinearRegression
m = 1000
n = 5000
big_x = np.random.normal(size=(m, n))
true_theta = np.random.uniform(0.0, 100.0, size=n+1)  # loc为均值，scale为标准差
big_y = big_x.dot(true_theta[1:]) + true_theta[0] + np.random.normal(loc=0.0, scale=10.0, size=m)
start = time.time()
big_reg1 = LinearRegression()
big_reg1.fit_normal(big_x, big_y)
end = time.time()
time1 = end - start
print('运行时间%fs' % time1)
start = time.time()
big_reg2 = LinearRegression()
big_reg2.fit_gd(big_x, big_y)
end = time.time()
time2 = end - start
print('运行时间%fs' % time2)
