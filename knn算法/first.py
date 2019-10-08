from cmath import log
import numpy as np
import operator


# 创建数据集
def creatDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 0, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算香农熵
def calcShannonent(dataSet):
    num_entyies = len(dataSet)
    labelCounts = {}
    for featvec in dataSet:
        currentLabel = featvec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        print(labelCounts)
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / num_entyies
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


# 划分数据
def splitDataSet(dataSet, axis, value):
    """

    :param dataSet: 待划分的数据集
    :param axis:划分数据集的特征
    :param value:返回值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:              # 判断axis列的值是否为value
            reducedFeatVec = featVec[:axis]     # [:axis]表示前axis列，即若axis为2，就是取featVec的前axis列
            print(reducedFeatVec)               # [axis+1:]表示从跳过axis+1行，取接下来的数据
            reducedFeatVec.extend(featVec[axis+1:])  # 列表扩展
            print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)
            print(retDataSet)
    return retDataSet


# 选择最优特征进行分离
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # the last column is used for the labels
    print("numFeatures:", numFeatures)
    baseEntropy = calcShannonent(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 计算每一特征对应的熵 ，然后:iterate over all the features
        featList = [example[i] for example in dataSet]
        # create a list of all the examples of this feature
        # print 'featList:', featList
        uniqueVals = set(featList)       # get a set of unique values
        # print(uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # print (subDataSet)
            prob = len(subDataSet)/float(len(dataSet))    # 计算子数据集在总的数据集中的比值
            newEntropy += prob * calcShannonent(subDataSet)
        # print(newEntropy)
        infoGain = baseEntropy - newEntropy
        # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       # compare this to the best gain so far
            bestInfoGain = infoGain         # if better than current best, set to best
            bestFeature = i
    return bestFeature
# 选出最优的特征，并返回特征角标 returns an integer


# 统计出现次数最多的分类名称
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 使用程序第二行导入运算符模块的itemgetter方法，按照第二个元素次序进行排序，逆序 ：从大到小
    return sortedClassCount[0][0]
# 创建决策树


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 抽取最优特征下的数值，重新组合成list，
    # print "featValues:", featValues
    uniqueVals = set(featValues)
    # print "uniqueVals:", uniqueVals
    for value in uniqueVals:
        subLabels = labels[:]       # copy all of labels, so trees don't mess up（搞错） existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        #print myTree
    return myTree


myDat, labels = creatDataSet()

print(calcShannonent(myDat))
myDat[0][-1] = 'maybe'
print(myDat, labels)
print('Ent changed: ', calcShannonent(myDat))
print('splitDataSet is :', splitDataSet(myDat, 1, 1))
print('the best feature is:', chooseBestFeatureToSplit(myDat))
mytree = createTree(myDat, labels)
print('createTree is ：', createTree(myDat, labels))
