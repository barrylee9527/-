import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(666)
x = np.random.uniform(-3, 3, size=100)
# x = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
plt.scatter(x, y)
plt.show()
x = x.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
train_score = []
test_score = []
for i in range(1, 76):
    lin_reg = LinearRegression()
    lin_reg.fit(x_train[:i], y_train[:i])
    y_train_predict = lin_reg.predict(x_train[:i])
    y_test_predict = lin_reg.predict(x_test)
    train_score.append(mean_squared_error(y_train[:i], y_train_predict))
    test_score.append(mean_squared_error(y_test, y_test_predict))
plt.plot([i for i in range(1, 76)], np.sqrt(train_score), label='train')
plt.plot([i for i in range(1, 76)], np.sqrt(test_score), label='test')
plt.legend()
plt.show()
def plot_learning_curve(algo, x_train, x_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(x_train) + 1):
        algo.fit(x_train[:i], y_train[:i])
        y_train_predict = algo.predict(x_train[:i])
        y_test_predict = algo.predict(x_test)
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))
        test_score.append(mean_squared_error(y_test, y_test_predict))
    plt.plot([i for i in range(1, len(x_train) + 1)], np.sqrt(train_score), label='train')
    plt.plot([i for i in range(1, len(x_train) + 1)], np.sqrt(test_score), label='test')
    plt.legend()
    plt.axis([0, len(x_train)+1, 0, 4])
    plt.show()
plot_learning_curve(LinearRegression(), x_train, x_test, y_train, y_test)
def PolynomialRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('lin_reg', LinearRegression())
    ])
poly2_reg = PolynomialRegression(degree=2)
plot_learning_curve(poly2_reg, x_train, x_test, y_train, y_test)



