import numpy as np
import operator
def createDataSet():
    dataSet = [[1, 1, 'yes'],  # 例如这个样本点代表不能浮出水面、有脚蹼，是鱼类
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']  # label记录样本的特征名称
    return dataSet, labels


def calcShannonEnt(dataSet):  # 输入数据集，计算信息熵
    numEntries = len(dataSet)  # 计算有多少个样本
    labelCounts = {}          # 创建一个字典，用于保存样本的标签，以及该标签对应的数量
    for featVec in dataSet:   # 遍历所有样本
        currentLabel = featVec[-1]   # 将每个样本的最后一列，即标签取出
        if currentLabel not in labelCounts.keys():  # 如果该标签不在字典中，就加入该标签，并且将数目置为0
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 将该标签对应的数量加1
    shannonEnt = 0.0                   # 初始化香农熵为0
    for key in labelCounts:            # 遍历字典，计算出最初的香农熵的值
        prob = float(labelCounts[key])/numEntries  # 计算每类标签所占的比重
        shannonEnt -= prob * np.log(prob, 2)  # 计算出香农熵，log的底为2
    return shannonEnt


def splitDataSet(dataSet, axis, value):  # 给定数据集、划分数据集的特征、特征所对应的值
    retDataSet = []                # 创建一个备份数据集，避免原始数据被修改
    for featVec in dataSet:        # 遍历数据集
        if featVec[axis] == value:  # 该特征维度下和value值相等的样本划分到一起，并将该特征去除掉维度去掉
            reducedFeatVec = featVec[:axis]    # 将axis维度两边的数据进行拼接，就将该特征维度给去除掉
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1  # 计算每个标签对应的数目
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 从大到小进行排列
    return sortedClassCount[0][0]  # 选出标签数量最多的返回


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # numfeature为特征的维度，因为最后一列为标签，所以需要减去1
    baseEntropy = calcShannonEnt(dataSet)  # 用来记录最小信息熵，初始值为原始数据集对应的信息熵
    bestInfoGain = 0.0; bestFeature = -1   # 信息增益初始化为0，最优的划分特征初始化为-1
    for i in range(numFeatures):           # 遍历所有的特征
        featList = [example[i] for example in dataSet]  # 创建list用来存每个样本在第i维度的特征值
        uniqueVals = set(featList)       # 获取该特征下的所有不同的值，即根据该特征可以划分为几类
        newEntropy = 0.0                 # 初始化熵为0
        for value in uniqueVals:         # 遍历该特征维度下对应的所有特征值
            subDataSet = splitDataSet(dataSet, i, value)  # 依据这个值，将样本划分为几个子集，有几个value，就有几个子集
            prob = len(subDataSet)/float(len(dataSet))   # 计算p值
            newEntropy += prob * calcShannonEnt(subDataSet)     # 计算每个子集对应的信息熵，并全部相加，得到划分后数据的信息熵
        infoGain = baseEntropy - newEntropy     # 将原数据的信息熵-划分后数据的信息熵，得到信息增益
        if (infoGain > bestInfoGain):      # 如果这个信息增益比当前记录的最佳信息增益还大，就将该增益和划分依据的特征记录下来
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature                      # returns an integer


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  # 存储所有样本的标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 如果所有的标签都是一样，就直接返回该子集的标签，这里用的方法是计算其中某个类别的标签数量，
        # 如果该数量就等于标签的总数，那容易知道，该数据集的类别标签是一样的
    if len(dataSet[0]) == 1:  # 如果样本的特征值就剩一个，即样本长度为1，就停止，返回该数据集标签数目最多，作为该数据集的标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选取出该数据集最佳的划分特征
    bestFeatLabel = labels[bestFeat]             # 得出该特征所对应的标签名称
    myTree = {bestFeatLabel:{}}                  # 创建mytree字典，这个字典将会一层套一层，这个看后面的结果就明白
    del(labels[bestFeat])  # 将这个最佳的划分特征从标签名称列表中删除，这是为了下次递归进来不会发生错误的引用，
    # 因为下面每次递归的数据集都会删除划分特征所对应的那一列
    featValues = [example[bestFeat] for example in dataSet]  # 获取到该划分特征下对应的所有特征值
    uniqueVals = set(featValues)  # 经过set去重，值代表着该特征能将当前的数据集划分成多少个不同的子集
    for value in uniqueVals:     # 现在对划分的子集进一步进行划分，也就是递归的开始
        subLabels = labels[:]       # 将样本标签复制给sublabels，这样就不会在每次的递归中改变原始labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  # 将样本划分的子集再进行迭代
    return myTree


madat, labels = createDataSet()
mytree = createTree(madat, labels)
print(mytree)
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree):
    numLeafs = 0
    # firstStr = myTree.keys()[0]
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 1
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
# firstStr = myTree.keys()[0] #注意这里和机器学习实战中代码不同，这里使用的是Python3，而在Python2中可以写成这种形式
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
mytree= retrieveTree(0)
createPlot(mytree)

