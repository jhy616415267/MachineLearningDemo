# _*_ coding: UTF-8 _*_

import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec




 # 创建全局文档
def createVocaList(dataSet):
    vacabSet = set([])
    for document in dataSet:
        vacabSet = vacabSet | set(document)
    return list(vacabSet)


def setofWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec




# 贝叶斯概率文档求解

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/numTrainDocs
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainDocs):
        if(trainCategory[i] == 1):
            p1Num += trainMatrix[i]
            p1Denom +=sum(trainMatrix[i])
        else:                                                #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive



def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1    			#对应元素相乘
    p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1
    else:
        return 0




def testingNB():
    listOPosts,listClasses = loadDataSet()									#创建实验样本
    myVocabList = createVocaList(listOPosts)								#创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setofWords2Vec(myVocabList, postinDoc))				#将实验样本向量化
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))		#训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']									#测试样本1
    thisDoc = np.array(setofWords2Vec(myVocabList, testEntry))				#测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')										#执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')										#执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']										#测试样本2

    thisDoc = np.array(setofWords2Vec(myVocabList, testEntry))				#测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')										#执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')										#执行分类并打印分类结果

if __name__ == '__main__':
    testingNB()


