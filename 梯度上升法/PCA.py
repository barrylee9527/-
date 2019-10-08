import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None

    def fit(self, x, eta=0.01, n_iters=1e4):
        def demean(x):
            return x - np.mean(x, axis=0)

        def f(w, x):
            return np.sum((x.dot(w) ** 2)) / len(x)

        def df_math(w, x):
            return x.T.dot(x.dot(w)) * 2. / len(x)

        def df(w, x):
            return x.T.dot(x.dot(w)) * 2. / len(x)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(x, initial_w, eta, n_iters=1e4, epsilon=1e-8):
            w = direction(initial_w)
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = df(w, x)
                last_w = w
                w = w + eta * gradient
                w = direction(w)  # 每次求一个单位方向
                if abs(f(w, x) - f(last_w, x)) < epsilon:
                    break
                cur_iter += 1
            return w

        x_pca = demean(x)
        self.components = np.empty(shape=(self.n_components, x.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(x_pca.shape[1])
            w = first_component(x_pca, initial_w, eta)
            self.components[i, :] = w
            x_pca = x_pca - x_pca.dot(w).reshape(-1, 1) * w
        return self

    def transform(self, x):
        assert x.shape[1] == self.components.shape[1]
        return x.dot(self.components.T)

    def inverse_transform(self, x):
        return x.dot(self.components)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components