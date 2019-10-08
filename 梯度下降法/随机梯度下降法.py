import numpy as np
import time
m = 100
x = np.random.normal(size=m)
y = 4. * x + 3. + np.random.normal(loc=0.0, scale=3.0, size=m)
x = x.reshape(-1, 1)


def dJ_sgd(theta, x_b_i, y_i):
    return x_b_i.T.dot(x_b_i.dot(theta) - y_i) * 2.


def sgd(x_b, y, initial_theta, n_iters):
    t0 = 5
    t1 = 50

    def learning_rate(t):
        return t0 / (t + t1)
    theta = initial_theta
    for cur_iter in range(n_iters):
        rand_i = np.random.randint(len(x_b))
        gradient = dJ_sgd(theta, x_b[rand_i], y[rand_i])
        theta = theta - learning_rate(cur_iter)*gradient
    return theta


start = time.time()
x_b = np.hstack([np.ones((len(x), 1)), x])
initial_theta = np.zeros(x_b.shape[1])
theta = sgd(x_b, y, initial_theta, n_iters=len(x_b) // 3)
end = time.time()
time1 = end - start
print('运行时间%fs' % time1)
print(theta[0], theta[1])


