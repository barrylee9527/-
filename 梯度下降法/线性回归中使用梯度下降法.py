import numpy as np
import matplotlib.pyplot as plt
np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3. + 4. + np.random.normal(size=100)
# plt.scatter(x, y)
# plt.show()
x = x.reshape(-1, 1)


def J(theta, x_b, y):
    try:
        return np.sum((y -x_b.dot(theta))**2)/len(x_b)
    except:
        return float('inf')


def dJ(theta, x_b, y):
    res = np.empty(len(theta))
    res[0] = np.sum(x_b.dot(theta) - y)
    for i in range(1, len(theta)):
        res[i] = (x_b.dot(theta) - y).dot(x_b[:, i])
    return res * 2 / len(x_b)


def gradint_descent(x_b, y, initial_theat, eta, n_iters = 1e4, epsilon=1e-8):
    theta = initial_theat
    i_iter = 0
    while i_iter < n_iters:
        gradient = dJ(theta, x_b, y)
        last_theat = theta
        theta = theta - eta * gradient
        if abs(J(theta,x_b, y) - J(last_theat, x_b, y)) < epsilon:
            break
        i_iter += 1
    return theta


x_b = np.hstack([np.ones((len(x), 1)), x])
# print(x_b)
# x_b = np.hstack((np.ones((len(x), 1)), x))
initial_theta = np.zeros(x_b.shape[1])
# print(initial_theta)
eta = 0.01
theta = gradint_descent(x_b, y, initial_theta, eta)
print(theta[0], theta[1])