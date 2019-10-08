import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 逻辑回归
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
log_score = log_clf.score(X_test, y_test)
# SVM
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_score = svm_clf.score(X_test, y_test)
# 决策树
dt_clf = DecisionTreeClassifier(random_state=666)
dt_clf.fit(X_train, y_train)
dt_score = dt_clf.score(X_test, y_test)
y_predict1 = log_clf.predict(X_test)
y_predict2 = svm_clf.predict(X_test)
y_predict3 = dt_clf.predict(X_test)
y_predict = np.array((y_predict1 + y_predict2 + y_predict3) >= 2, dtype='int')
score1 = accuracy_score(y_test, y_predict)
print(score1)
# 使用Voting Classifier  少数服从多数
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=666))], voting='hard')
voting_clf.fit(X_train, y_train)
score2 = voting_clf.score(X_test, y_test)
print(score2)
# 更合理的投票，应该有权值
# Hard Voting Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=666))],
                             voting='hard')
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
# 使用 Soft Voting Classifier
voting_clf2 = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC(probability=True)),
    ('dt_clf', DecisionTreeClassifier(random_state=666))],
                             voting='soft')
voting_clf2.fit(X_train, y_train)
voting_clf2.score(X_test, y_test)
