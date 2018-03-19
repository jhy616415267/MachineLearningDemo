# _*_ coding: UTF-8 _*_


import operator

import numpy as np


# 创建数据集

def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])

    labels = ['爱情片', '爱情片', '动作片', '动作片']

    return group, labels


# KNN算法
def classfy0(testData, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    diffMat = np.tile(testData, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat ** 2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances ** 0.5

    sortedDistIndices = distances.argsort()

    print(sortedDistIndices)

    classCount = {}

    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]

        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

    print (dataSetSize)


if __name__ == '__main__':
    group, labels = createDataSet()
    print (group)
    print (labels)
    test = [101, 20]
    testClass = classfy0(test, group, labels, 3)

    print(testClass)
