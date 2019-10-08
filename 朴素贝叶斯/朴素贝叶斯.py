from numpy import *
import numpy as np
from functools import reduce

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):       # 创建一个空的不重复列表
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets   # 取并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)    # 创建一个其中所含元素都为0的向量
    for word in inputSet:                   # 遍历每个词条
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1      # 如果词条存在于词汇表中，则置1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec     # 返回文档向量


wordtxt, labellist = loadDataSet()
wordtext = [['hello', 'dog', 'has', 'flea', 'problem', 'help', 'please']]
wordlist = createVocabList(wordtxt)
np.array(wordlist)
print(wordlist, len(wordlist))
arr = np.array(setOfWords2Vec(wordlist, wordtext[0]))
print(wordtxt[0])
print(arr, len(arr))


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 文档的总数
    numWords = len(trainMatrix[0])  # 单词的总数，每一个文档的词向量长度都等于字典的长度
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 侮辱性文档出现概率，与文档的总数相除就得到了侮辱性文件的出现概率,即p(c1)
    p0Num = np.zeros(numWords)  # [0,0,0,.....]   # 构造非侮辱性文档中单词出现次数列表，列表长度和字典长度一致
    p1Num = np.zeros(numWords)  # [0,0,0,.....]   # 构造侮辱性文档中单词出现次数列表，列表长度和字典长度一致

    p0Denom = 0.0  # 记录非侮辱性文档的单词总数
    p1Denom = 0.0  # 记录侮辱性文档的单词总数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 遍历所有文档
            # 如果是侮辱性文档，对侮辱性文档的向量进行加和
            p1Num += trainMatrix[i]  # [0,1,1,....] + [0,1,1,....]->[0,2,2,...]
            # 对向量中的所有元素进行求和，也就是计算所有侮辱性文档中出现的单词总数
            p1Denom += sum(trainMatrix[i])
        else:  # 如果是非侮辱性文档，则执行如下，逻辑和上面一样
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom  # [1,2,3,5]/90->[1/90,...] # 在侮辱性类别下，每个单词出现的概率，即p(wi|c1)
    p0Vect = p0Num / p0Denom  # 在非侮辱性类别下，每个单词出现的概率，即p(wi|c0)
    return p0Vect, p1Vect, pAbusive

# 改进，解决0概率问题和数据下溢出，小数乘小数越乘越小，最后会导致变0
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                            # 计算训练的文档数目
    numWords = len(trainMatrix[0])                            # 计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)        # 文档属于侮辱类的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)    # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0; p1Denom = 2.0                            # 分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                            # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                                                # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)                            # 取对数，防止下溢出
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
print('p0V:\n', p0V)
print('p1V:\n', p1V)
print('classVec:\n', listClasses)
print('pAb:\n', pAb)
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1    			#对应元素相乘
	p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
	print('p0:',p0)
	print('p1:',p1)
	if p1 > p0:
		return 1
	else:
		return 0
# 改进之后的
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0



def testingNB():
    listOPosts,listClasses = loadDataSet()     #创建实验样本
    myVocabList = createVocabList(listOPosts)  #创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))    #将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))    #训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']  # 输入样本
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']  # 输入样本
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
testingNB()


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

