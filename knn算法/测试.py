import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
iris.keys()
x = iris.data
y = iris.target
shuffled_indexes = np.random.permutation(len(x))
test_ratio = 0.2
test_size = int(len(x) * test_ratio)
test_indexes = shuffled_indexes[:test_size]
train_indexes = shuffled_indexes[test_size:]
x_train = x[train_indexes]
y_train = y[train_indexes]

x_test = x[test_indexes]
y_test = y[test_indexes]
print(x_test.shape, y_test.shape)