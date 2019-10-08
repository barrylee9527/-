from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
boston = datasets.load_boston()
x = boston.data
y = boston.target
x = x[y < 50.0]
y = y[y < 50.0]
# x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
print(lin_reg.coef_)
print(lin_reg.intercept_)
score = lin_reg.score(x_test, y_test)
print(score)
