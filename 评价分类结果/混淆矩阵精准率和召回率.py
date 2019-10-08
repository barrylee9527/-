import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
digits = datasets.load_digits()
x = digits.data
y = digits.target.copy()
y[digits.target == 9] = 1
y[digits.target == 9] = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
score = log_reg.score(x_test, y_test)
print(score)
y_log_predict = log_reg.predict(x_test)
def TN(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 0))
def FP(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 1))
def FN(y_true, y_predict):
    return np.sum((y_true == 1) & (y_predict == 0))
def TP(y_true, y_predict):
    return np.sum((y_true == 1) & (y_predict == 1))
tn = TN(y_test, y_log_predict)
print(tn)
fp = FP(y_test, y_log_predict)
print(fp)
fn = FN(y_test, y_log_predict)
print(fn)
tp = TP(y_test, y_log_predict)
print(tp)
def confusion_matrix(y_true, y_predict):
    return np.array([
       [TN(y_true, y_predict), FP(y_true, y_predict)],
       [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])
con_mat = confusion_matrix(y_test, y_log_predict)
print(con_mat)
def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp /(tp+fp)
    except:
        return 0.0
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp /(tp+fn)
    except:
        return 0.0
precision = precision_score(y_test, y_log_predict)
recall = recall_score(y_test, y_log_predict)
def f1_score(precision, recall):
    try:
        return 2 * precision * recall / (precision + recall)
    except Exception as e:
        return repr(e)
f1 = f1_score(precision, recall)
print(precision, recall, f1)
# sklearn中的
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precision = precision_score(y_test, y_log_predict)
recall = recall_score(y_test, y_log_predict)
print(precision, recall)
dec_score = log_reg.decision_function(x_test)  # 决策的分数值
print(np.max(dec_score), np.min(dec_score))
y_predict2 = np.array(dec_score >= 5, dtype='int')
# 可视化方式观察
precisions = []
recalls = []
thresholds = np.arange(np.min(dec_score), np.max(dec_score), 0.1)
for threshold in thresholds:
    y_predict = np.array(dec_score >= threshold, dtype='int')
    precisions.append(precision_score(y_test, y_predict))
    recalls.append(recall_score(y_test, y_predict))
plt.plot(thresholds, precisions)
plt.plot(thresholds, recalls)
plt.show()
plt.plot(precisions, recalls)
plt.show()
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, dec_score)
def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0
def TFR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.0
# ROC曲线

fprs = []
tprs = []
thresholds = np.arange(np.min(dec_score), np.max(dec_score), 0.1)
for threshold in thresholds:
    y_predict = np.array(dec_score >= threshold, dtype='int')
    fprs.append(precision_score(y_test, y_predict))
    tprs.append(recall_score(y_test, y_predict))
plt.plot(fprs, tprs)
plt.show()
from sklearn.metrics import roc_curve
fprs, tprs, thresholds = roc_curve(y_test, dec_score)
plt.plot(fprs, tprs)
plt.show()
# 曲线下面的面积
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, dec_score)
