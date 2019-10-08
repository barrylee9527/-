import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
iris = datasets.load_iris()  # 鸢尾花数据集
x = iris.data[:, 2:]
y = iris.target
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.scatter(x[y == 2, 0], x[y == 2, 1])
plt.show()  # max_depth 最大深度 criterion熵
dt_clf = DecisionTreeClassifier(max_depth=2, criterion='entropy')
dt_clf.fit(x, y)
score = dt_clf.score(x, y)
print(score)


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100))
    )
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(x_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.scatter(x[y == 2, 0], x[y == 2, 1])
plt.show()
# 模拟使用信息熵进行划分


def split(x, y, d, value):
    index_a = (x[:, d] <= value)
    index_b = (x[:, d] > value)
    return x[index_a], x[index_b], y[index_a], y[index_b]


from collections import Counter


def entropy(y):
    counter = Counter(y)  # 编成字典格式
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p * np.log(p)
        return res


def try_split(x, y):
    best_entropy = float('inf')
    best_d, best_v = -1, 1
    for d in range(x.shape[1]):
        sorted_index = np.argsort(x[:, d])
        for i in range(1, len(x)):
            if x[sorted_index[i-1], d] != x[sorted_index[i], d]:
                v = (x[sorted_index[i-1], d] + x[sorted_index[i], d]) / 2
                x_1, x_r, y_1, y_r = split(x, y, d, v)
                e = entropy(y_1) + entropy(y_r)
                if e < best_entropy:
                    best_entropy, best_d, best_v = e, d, v
    return best_entropy, best_d, best_v


best_entropy, best_d, best_v = try_split(x, y)
print(best_entropy, best_d, best_v)


