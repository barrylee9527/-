import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_text = train_test_split(x, y, test_size=0.2, random_state=666)
standardscaler = StandardScaler()
standardscaler.fit(x_train)
print(standardscaler.mean_)
x_train = standardscaler.transform(x_train)
x_test = standardscaler.transform(x_test)
knn_clf = KNeighborsClassifier(3)
knn_clf.fit(x_train, y_train)
score = knn_clf.score(x_test, y_text)
print(score)