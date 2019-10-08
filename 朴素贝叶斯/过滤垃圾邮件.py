import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


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


listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)      # 生成全1矩阵，因为0无法进行log计算
    p0Denom = 2.0
    p1Denom = 2.0                        # change to 2.0
    for i in range(numTrainDocs):  # 遍历所有的文档
        if trainCategory[i] == 1:  # 如果是侮辱性文档，就
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          # change to log()
    p0Vect = np.log(p0Num/p0Denom)          # change to log()
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']  # 输入样本
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']  # 输入样本
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOPosts if len(tok) > 2]


def spamTest():
    # 新建三个列表
    docList = []
    classList = []
    fullTest = []
    # i 由1到25
    for i in range(1, 6):
        # 打开并读取指定目录下的本文中的长字符串，并进行处理返回
        wordList = testParse(open('E:\邮件/%d.txt' % i).read())
        # 将得到的字符串列表添加到docList
        docList.append(wordList)
        # 将字符串列表中的元素添加到fullTest
        fullTest.extend(wordList)
        # 类列表添加标签1
        classList.append(1)
        # 打开并取得另外一个类别为0的文件，然后进行处理
        wordList = testParse(open('email/ham/&d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    # 将所有邮件中出现的字符串构建成字符串列表
    vocabList = createVocabList(docList)
    # 构建一个大小为50的整数列表和一个空列表
    trainingSet = range(50)
    testSet = []
    # 随机选取1~50中的10个数，作为索引，构建测试集
    for i in range(10):
        # 随机选取1~50中的一个整型数
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        # 将选出的数的列表索引值添加到testSet列表中
        testSet.append(trainingSet[randIndex])
        # 从整数列表中删除选出的数，防止下次再次选出
        # 同时将剩下的作为训练集
        del (trainingSet[randIndex])
    # 新建两个列表
    trainMat = []
    trainClasses = []
    # 遍历训练集中的每个字符串列表
    for docIndex in trainingSet:
        # 将字符串列表转为词条向量，然后添加到训练矩阵中
        trainMat.append(setOfWords2Vec(vocabList, fullTest[docIndex]))
        # 将该邮件的类标签存入训练类标签列表中
        trainClasses.append(classList[docIndex])
    # 计算贝叶斯函数需要的概率值并返回
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    # 遍历测试集中的字符串列表
    for docIndex in testSet:
        # 同样将测试集中的字符串列表转为词条向量
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 对测试集中字符串向量进行预测分类，分类结果不等于实际结果
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
        print('the error rate is:', float(errorCount) / len(testSet))
spamTest()