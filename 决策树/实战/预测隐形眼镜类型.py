"""
一共有24组数据，数据的Labels依次是age、prescript、astigmatic、tearRate、class，
也就是第一列是年龄，第二列是症状，第三列是是否散光，第四列是眼泪数量，第五列是最终的分类标签
隐形眼镜类型包括硬材质(hard)、软材质(soft)以及不适合佩戴隐形眼镜(no lenses)。
"""
import os
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder
d = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
filename = os.path.join(d, 'lenses.txt')
print(filename)
fr = open(filename, 'r')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# print(lenses)
lenses_target = []                                                        # 提取每组数据的类别，保存在列表里
for each in lenses:
    lenses_target.append(each[-1])
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征标签
lenses_list = []  # 保存lenses数据的临时列表
lenses_dict = {}  # 保存lenses数据的字典，用于生成pandas
for each_label in lensesLabels:  # 提取信息，生成字典
    for each in lenses:
        lenses_list.append(each[lensesLabels.index(each_label)])
    lenses_dict[each_label] = lenses_list
    lenses_list = []
# print(lenses_dict)  # 打印字典信息
lenses_pd = pd.DataFrame(lenses_dict)  # 生成pandas.DataFrame
print(lenses_pd)
le = LabelEncoder()                                                        # 创建LabelEncoder()对象，用于序列化
for col in lenses_pd.columns:  # 为每一列序列化
    lenses_pd[col] = le.fit_transform(lenses_pd[col])
print(lenses_pd)
des = DecisionTreeClassifier()
des.fit(lenses_pd.values.tolist(), lenses_target)
print(des.predict([[1, 1, 1, 0]])[0])
print(des.score(lenses_pd.values.tolist(), lenses_target))




