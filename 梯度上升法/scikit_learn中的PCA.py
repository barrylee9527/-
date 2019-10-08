"""from sklearn.decomposition import PCA
import numpy as np
x = np.empty((100, 2))
x[:, 0] = np.random.uniform(0., 100., size=100)
x[:, 1] = 0.75 * x[:, 0] + 3. + np.random.normal(0, 10., size=100)
pca = PCA(n_components=1)
pca.fit(x)
print(pca.components_)
x_reduction = pca.transform(x)
print(x_reduction.shape)"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
digits = datasets.load_digits()
x = digits.data
print(x.shape)
y = digits.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)
score = knn_clf.score(x_test, y_test)
print(score)
pca = PCA(n_components=50)
pca.fit(x_train)
x_train_reduction = pca.transform(x_train)
x_test_reduction = pca.transform(x_test)
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_reduction, y_train)
score = knn_clf.score(x_test_reduction, y_test)
print(score)
print(pca.explained_variance_ratio_)  # 解释的方差比例
pca = PCA(n_components=x_train.shape[1])
pca.fit(x_train)
print(pca.explained_variance_ratio_)
plt.plot([i for i in range(x_train.shape[1])], [np.sum(pca.explained_variance_ratio_[: i+1]) for i in range(x_train.shape[1])])
plt.show()
pca = PCA(0.95)
pca.fit(x_train)
print(pca.n_components_)
x_train_reduction = pca.transform(x_train)
x_test_reduction = pca.transform(x_test)
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_reduction, y_train)
score = knn_clf.score(x_test_reduction, y_test)
print(score)
pca = PCA(n_components=2)
pca.fit(x)
x_reduction = pca.transform(x)
for i in range(10):
    plt.scatter(x_reduction[y == i, 0], x_reduction[y == i, 1], alpha=0.8)
plt.show()
