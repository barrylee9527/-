import numpy as np
import matplotlib.pyplot as plt


x = np.empty((100, 2))
x[:, 0] = np.random.uniform(0., 100., size=100)
x[:, 1] = 0.75 * x[:, 0] + 3. + np.random.normal(0, 10., size=100)
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()


def demean(x):
    return x - np.mean(x, axis=0)   # axis代表行方向


x_demean = demean(x)
# plt.scatter(x_deman[:, 0], x_deman[:, 1])
# plt.show()
# 梯度上升法


def f(w, x):
    return np.sum((x.dot(w)**2))/len(x)


def df(w, x):
    return x.T.dot(x.dot(w))*2./len(x)


def direction(w):
    return w / np.linalg.norm(w)  # 模


def first_component(x, initial_w, eta, n_iters= 1e4, epsilon=1e-8):
    w = direction(initial_w)
    cur_iter = 0
    while cur_iter < n_iters:
        gradient = df(w, x)
        last_w = w
        w = w + eta*gradient
        w = direction(w)  # 每次求一个单位方向
        if abs(f(w, x) - f(last_w, x)) < epsilon:
            break
        cur_iter += 1
    return w


initial_w = np.random.random(x.shape[1])  # 不能用0向量开始
print(initial_w)
eta = 0.001
w = first_component(x_demean, initial_w, eta)
print(w)
x2 = np.empty(x.shape)
print(len(x))
for i in range(len(x)):
    x2[i] = x[i] - x[i].dot(w)*w
plt.scatter(x2[:, 0], x2[:, 1])
plt.show()
w2 = first_component(x2, initial_w, eta)
print(w.dot(w2))


def first_n_components(n, x, eta=0.01, n_iters=1e4, epsilon=1e-8):
    x_pca = x.copy()
    x_pca = demean(x_pca)
    res = []
    for i in range(n):
        initial_w = np.random.random(x_pca.shape[1])
        w = first_component(x_pca, initial_w, eta)
        res.append(w)
        x_pca = x_pca - x_pca.dot(w).reshape(-1, 1)*w
    return res
print(first_n_components(2, x))