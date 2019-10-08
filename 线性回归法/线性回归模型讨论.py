from sklearn import datasets
from sklearn.linear_model import LinearRegression
import numpy as np
boston = datasets.load_boston()
x = boston.data
y = boston.target
x = x[y < 50.0]
y = y[y < 50.0]
lin_reg = LinearRegression()
lin_reg.fit(x, y)
# print(lin_reg.coef_)
arg_sort = np.argsort(lin_reg.coef_)  # 排序序列
feature_name = boston.feature_names[arg_sort]  # feature_namde为数组
# print(feature_name)
print(boston.DESCR)

