from numpy import *
from os import listdir
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    r = zeros((1,1024))
    with open(filename, 'r') as file:
        for i in range(32):
            line = file.readline()
            for j in range(32):
                r[0,32*i+j] = int(line[j])
    return r

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fn = trainingFileList[i]
        classNumStr = int(fn.split('.')[0].split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("trainingDigits/{}".format(fn))
    testFileList = listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fn = testFileList[i]
        classNumStr = int(fn.split('.')[0].split('_')[0])
        vectorUnderTest = img2vector("testDigits/{}".format(fn))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("Classifier: {}\tReal answer: {}".format(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("Errors: {}\nError Rate: {}".format(errorCount, errorCount/float(mTest)))
