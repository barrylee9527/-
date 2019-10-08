from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np
mnist = fetch_mldata('MNIST original')
x, y = mnist['data'], mnist['target']
x_train = np.array(x[:60000], dtype=float)
y_train = np.array(x[:60000], dtype=float)
x_test = np.array(x[:60000], dtype=float)
y_test = np.array(x[:60000], dtype=float)
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)
# PCA降维
pca = PCA(0.9)
pca.fit(x_train)
x_train_reduction = pca.transform(x_train)
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_reduction, y_train)
x_test_reduction = pca.transform(x_test)
score = knn_clf.score(x_test_reduction, y_test)
print(score)

