import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
boston = datasets.load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
def StandardLinearSVR(epsion=0.1):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('linearSVC', LinearSVR(epsilon=epsion))
    ]
    )
svr = StandardLinearSVR()
svr.fit(x_train, y_train)
score = svr.score(x_test, y_test)
print(score)