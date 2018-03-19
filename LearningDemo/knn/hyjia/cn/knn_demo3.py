# _*_ coding: UTF-8 _*_

from os import listdir

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN


# 将32*32的二进制图像转换为1*1024向量

def img2Vector(filename):
    returnVect = np.zeros((1, 1024))

    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])

    return returnVect


# 手写数字分类测试
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')

    m = len(trainingFileList)

    trainingMat = np.zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]

        classNumber = int(fileNameStr.split('_')[0])

        hwLabels.append(classNumber)

        trainingMat[i, :] = img2Vector('trainingDigits/%s' % (fileNameStr))

    neigh = kNN(n_neighbors=3, algorithm='auto')

    neigh.fit(trainingMat, hwLabels)

    testFileList = listdir('testDigits')

    errorCount = 0.0

    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]

        classNumber = int(fileNameStr.split('_')[0])

        vectorUnderTest = img2Vector('testDigits/%s' % (fileNameStr))

        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0

    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
       handwritingClassTest()
