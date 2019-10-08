import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
score = log_reg.score(x_test, y_test)
y_predict = log_reg.predict(x_test)
from sklearn.metrics import precision_score
precision_score(y_test, y_predict, average='micro')  # 多分类
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_predict))
cfm = confusion_matrix(y_test, y_predict)
plt.matshow(cfm, cmap=plt.cm.gray)
plt.show()
row_sums = np.sum(cfm, axis=1)
err_metrix = cfm / row_sums
np.fill_diagonal(err_metrix, 0)
print(err_metrix)
plt.matshow(err_metrix, cmap=plt.cm.gray)
plt.show()




