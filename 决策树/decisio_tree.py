import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
x = iris.data
y = iris.target
x = iris.data[y < 2, :2]
y = iris.target[y < 2]
print(x.shape, y.shape)
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.show()
standard_scaler = StandardScaler()
standard_scaler.fit(x)
x_standard = standard_scaler.transform(x)
svc = LinearSVC(C=1e9)
svc.fit(x_standard, y)
