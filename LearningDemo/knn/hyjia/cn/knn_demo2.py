# _*_ coding: UTF-8 _*_


import operator

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


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


# 解析数据
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberoflines = len(arrayOlines)
    returnMat = np.zeros((numberoflines, 3))
    classlableVector = []

    index = 0

    for line in arrayOlines:
        line = line.strip()

        listFromLine = line.split('\t')

        returnMat[index, :] = listFromLine[0:3]

        if listFromLine[-1] == 'didntLike':
            classlableVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classlableVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classlableVector.append(3)
        index += 1

    return returnMat, classlableVector


# 数据集可视化

def showdatas(datingDataMat, datingLabels):
    font = FontProperties()
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()

    # 对数据进行归一化处理


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每列的最小值
    maxVals = dataSet.max(0)  # 每列的最大值

    ranges = maxVals - minVals

    normDataSet = np.zeros(np.shape(dataSet))

    m = dataSet.shape[0]

    normDataSet = dataSet - np.tile(minVals, (m, 1))

    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

    # 测试


def datingTestClass():
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)

    print (datingDataMat)
    print(datingLabels)
    #取所有数据的百分之十
    hoRatio = 0.10
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获得normMat的行数
    m = normMat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    #分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classfy0(normMat[i,:], normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))
    showdatas(datingDataMat, datingLabels)


if __name__ == '__main__':
    datingTestClass()
