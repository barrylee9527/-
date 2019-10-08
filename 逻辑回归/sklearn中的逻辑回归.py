import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
np.random.seed(666)
x = np.random.normal(0, 1, size=(200, 2))
y = np.array(x[:, 0]**2 + x[:, 1] < 1.5, dtype='int')
for _ in range(20):
    y[np.random.randint(200)] = 1
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
score1 = log_reg.score(x_train, y_train)
score2 = log_reg.score(x_test, y_test)
print(score1, score2)
def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('lin_reg', LogisticRegression())
    ])
poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(x_train, y_train)
score3 = poly_log_reg.score(x_test, y_test)
score4 = poly_log_reg.score(x_train, y_train)
print(score3, score4)
def PolynomialLogisticRegression(degree, C):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('lin_reg', LogisticRegression(C=C))
    ])
poly_log_reg2 = PolynomialLogisticRegression(degree=2, C=0.1)
poly_log_reg2.fit(x_train, y_train)
score5 = poly_log_reg2.score(x_test, y_test)
score6 = poly_log_reg2.score(x_train, y_train)
print(score5, score6)
def PolynomialLogisticRegression(degree, C, penalty='12'):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('lin_reg', LogisticRegression(C=C, penalty=penalty))
    ])
poly_log_reg3 = PolynomialLogisticRegression(degree=2, C=0.1, penalty='11')
poly_log_reg3.fit(x_train, y_train)
score7 = poly_log_reg3.score(x_test, y_test)
score8 = poly_log_reg3.score(x_train, y_train)
print(score7, score8)
