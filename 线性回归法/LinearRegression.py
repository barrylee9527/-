import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, x_train, y_train):
        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, x_predict):
        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        return x_b.dot(self._theta)

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        mse_test = np.sum((y_predict - y_test) ** 2) / len(y_test)
        var_test = np.sum((np.mean(y_test) - y_test)**2)
        return 1 - mse_test * len(y_test)/var_test

    def __repr__(self):
        return "LinearRegression()"

    def fit_gd(self, x_train, y_train, eta=0.01, n_iters=1e4):

        def J(theta, x_b, y):
            try:
                return np.sum((y - x_b.dot(theta)) ** 2) / len(x_b)
            except:
                return float('inf')

        def dJ(theta, x_b, y):
            """res = np.empty(len(theta))
            res[0] = np.sum(x_b.dot(theta) - y)
            for i in range(len(theta)):
                res[i] = (x_b.dot(theta) - y).dot(x_b[:, 1])
            return res * 2 / len(x_b)"""
            return x_b.T.dot(x_b.dot(theta) - y) * 2. / len(x_b)

        def gradint_descent(x_b, y, initial_theat, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theat
            i_iter = 0
            while i_iter < n_iters:
                gradient = dJ(theta, x_b, y)
                last_theat = theta
                theta = theta - eta * gradient
                if (abs(J(theta, x_b, y) - J(last_theat, x_b, y)) < epsilon):
                    break
                i_iter += 1
            return theta
        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = gradint_descent(x_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_sgd(self, x_train, y_train, n_iters=1e4, t0 = 5, t1 = 50):
        def dJ_sgd(theta, x_b_i, y_i):
            return x_b_i.T.dot(x_b_i.dot(theta) - y_i) * 2.

        def sgd(x_b, y, initial_theta, n_iters, t0=5, t1=50):
            def learning_rate(t):
                return t0 / (t + t1)
            theta = initial_theta
            m = len(x_b)
            for cur_iter in range(n_iters*m):
                indexs = np.random.permutation(m)  # 随机排列
                x_b_new = x_b[indexs]
                y_new = y[indexs]
                for i in range(m):
                    gradient = dJ_sgd(theta, x_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter*m + i) * gradient
            return theta

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.random.randn(x_b.shape[1])
        self._theta = sgd(x_b, y_train, initial_theta, n_iters, t0, t1)
        self._intercept = self._theta[0]
        self.coef_ = self._theta[1:]



