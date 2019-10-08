import numpy as np
import matplotlib.pyplot as plt
np.random.seed(666)
x = np.random.random(size=(1000, 10))
true_theta = np.arange(1, 12, dtype=float)
x_b = np.hstack([np.ones((len(x), 1)), x])
y = x_b.dot(true_theta) + np.random.normal(size=1000)


def J(theta, x_b, y):     # 代价函数
    try:
        return np.sum((y - x_b.dot(theta))**2)/len(x_b)
    except:
        return float('inf')


def dJ_math(theta, x_b, y):   # 代价函数的梯度
    return x_b.T.dot(x_b.dot(theta) - y) * 2. / len(y)


def dJ_debug(theta, x_b, y, epsilon=0.01):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2[i] -= epsilon
        res[i] = (J(theta_1, x_b, y) - J(theta_2, x_b, y)) / (2*epsilon)
        return res


def gradint_descent(dJ, x_b, y, initial_theat, eta, n_iters = 1e4, epsilon=1e-8):  # 梯度下降的迭代
    theta = initial_theat
    i_iter = 0
    while i_iter < n_iters:
        gradient = dJ(theta, x_b, y)
        last_theat = theta
        theta = theta - eta * gradient
        if(abs(J(theta,x_b, y) - J(last_theat, x_b, y)) < epsilon):
            break
        i_iter += 1
    return theta
x_b = np.hstack([np.ones((len(x), 1)), x])
initial_theta = np.zeros(x_b.shape[1])
eta = 0.01
theta = gradint_descent(dJ_debug, x_b, y, initial_theta, eta)
print(theta)

