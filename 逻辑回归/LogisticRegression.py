import numpy as np


class LogisticRegression:
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None
        
    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def predict(self, x_predict):
        proba = self.predict_proba(x_predict)
        return np.array(proba >= 0.5, dtype='int')

    def predict_proba(self, x_predict):
        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        return self._sigmoid(x_b.dot(self._theta))

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        mse_test = np.sum((y_predict - y_test) ** 2) / len(y_test)
        var_test = np.sum((np.mean(y_test) - y_test)**2)
        return 1 - mse_test * len(y_test)/var_test

    def __repr__(self):
        return "LogisticRegression()"

    def fit(self, x_train, y_train, eta=0.01, n_iters=1e4):
        def J(theta, x_b, y):
            y_hat = self._sigmoid(x_b.dot(theta))
            try:
                return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(theta, x_b, y):
            """res = np.empty(len(theta))
            res[0] = np.sum(x_b.dot(theta) - y)
            for i in range(len(theta)):
                res[i] = (x_b.dot(theta) - y).dot(x_b[:, 1])
            return res * 2 / len(x_b)"""
            return x_b.T.dot(self._sigmoid(x_b.dot(theta)) - y) / len(x_b)

        def gradint_descent(x_b, y, initial_theat, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theat
            i_iter = 0
            while i_iter < n_iters:
                gradient = dJ(theta, x_b, y)
                last_theat = theta
                theta = theta - eta * gradient
                if abs(J(theta, x_b, y) - J(last_theat, x_b, y)) < epsilon:
                    break
                i_iter += 1
            return theta
        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = gradint_descent(x_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self


