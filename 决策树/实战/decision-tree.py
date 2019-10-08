"""
年龄：0代表青年，1代表中年，2代表老年；
有工作：0代表否，1代表是；
有自己的房子：0代表否，1代表是；
信贷情况：0代表一般，1代表好，2代表非常好；
类别(是否给贷款)：no代表否，yes代表是。

"""
import numpy as np
import cmath
import operator
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder

def CreateDatasets():
    datasets = [[0, 0, 0, 0, 'no'],         # 数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return datasets, labels


def CalculateShanno(datasets):
    numEntires = len(datasets)
    labelCounts = {}
    for vec in datasets:
        currentLabel = vec[-1]  # 提取标签(Label)信息
        # print(currentLabel)
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # print(labelCounts)
    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        # print(key)
        prob = float(labelCounts[key]) / numEntires  # 选择该标签(Label)的概率
        # print(prob)
        shannonEnt -= prob * cmath.log(prob, 2)  # 利用公式计算
        # print(shannonEnt)
    return shannonEnt  # 返回经验熵(香农熵)


def splitDataSet(dataSet, axis, value):
    retDataSet = []                                     # 创建返回的数据集列表
    for featVec in dataSet:                             # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]             # 去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])     # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet                                    # 返回划分后的数据集


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # print(numFeatures)
    # 特征数量
    baseEntropy = CalculateShanno(dataSet)               # 计算数据集的香农熵
    bestInfoGain = 0.0                                   # 信息增益
    bestFeature = -1                                     # 最优特征的索引值
    for i in range(numFeatures):                         # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                       # 创建set集合{},元素不可重复
        newEntropy = 0.0                                 # 经验条件熵
        # print(uniqueVals)
        for value in uniqueVals:                         # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)        # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))        # 计算子集的概率
            newEntropy += prob * CalculateShanno(subDataSet)    # 根据公式计算经验条件熵
        infoGain = (baseEntropy - newEntropy).real              # 信息增益  .real是返回复数的实部
        print(infoGain)
        print("第%d个特征的增益为%.3f" % (i, infoGain))          # 打印每个特征的信息增益
        if (infoGain > bestInfoGain):                           # 计算信息增益
            bestInfoGain = infoGain                             # 更新信息增益，找到最大的信息增益
            bestFeature = i                                     # 记录信息增益最大的特征的索引值
        # print(featList)
    return bestFeature
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        #根据字典的值降序排序
    return sortedClassCount[0][0]                                #返回classList中出现次数最多的元素
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]            #取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):            #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                                    #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                #选择最优特征
    bestFeatLabel = labels[bestFeat]                            #最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                                    #根据最优特征的标签生成树
    del(labels[bestFeat])                                        #删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]        #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                #去掉重复的属性值
    for value in uniqueVals:                                    #遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))                                                        #获取决策树结点
    secondDict = inputTree[firstStr]                                                        #下一个字典
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
# 存储决策树
def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)
datasets, labels = CreateDatasets()
featLabels = []
myTree = createTree(datasets, labels, featLabels)
print(myTree)
testVec = [0, 1]   # 其中的0代表没工作，1代表有房子                                   # 测试数据
result = classify(myTree, featLabels, testVec)
if result == 'yes':
    print('放贷')
if result == 'no':
    print('不放贷')
storeTree(myTree, 'classifierStorage.txt')
