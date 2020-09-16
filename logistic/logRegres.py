from numpy import *
import matplotlib.pyplot as plt
import random

def loadDataSet():
    data,labels = [],[]
    with open('testSet.txt') as f:
        for line in f:
            vals = line.split()
            data.append([1.0, float(vals[0]), float(vals[1])])
            labels.append(int(vals[2]))
    return data, labels

def sigmoid(x):
    return 1.0/(1+exp(-x))

def gradAscent(data, labels):
    data,labels = mat(data), mat(labels).transpose()
    m,n = shape(data)
    alpha,maxCycles = 0.001, 500
    weights = ones((n,1))
    for _ in range(maxCycles):
        s = sigmoid(data*weights)
        error = labels-s
        weights = weights + alpha * data.transpose() * error
    return weights

def stocGradAscent0(data, labels):
    m,n = shape(data)
    alpha,weights = 0.01, ones(n)
    for i in range(m):
        h = sigmoid(sum(data[i]*weights))
        error = labels[i] - h
        weights = weights + alpha * error * data[i]
    return weights

def stocGradAscent1(data, labels, numIter=150):
    m,n = shape(data)
    weights = ones(n)
    idx = list(range(m))
    for k1 in range(numIter):
        random.shuffle(idx)
        for k2,i in enumerate(idx):
            alpha = 4/(1.0+k1+k2)+0.01
            h = sigmoid(sum(data[i]*weights))
            error = labels[i] - h
            weights = weights + alpha * error * data[i]
    return weights

def plotBestFit(weights):
    data,labels = loadDataSet()
    data = array(data)
    n = shape(data)[0]
    x1,y1,x2,y2 = [],[],[],[]
    for i in range(n):
        if int(labels[i]) == 1:
            x1.append(data[i,1])
            y1.append(data[i,2])
        else:
            x2.append(data[i,1])
            y2.append(data[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1,y1, s=30, c='red', marker='s')
    ax.scatter(x2,y2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classifyVector(x, weights):
    p = sigmoid(sum(x * weights))
    return 1.0 if p>0.5 else 0.0

def colicTest():
    data,labels = [],[]
    for line in open("horseColicTraining.txt"):
        vals = list(map(float, line.split()))
        data.append(vals[:21])
        labels.append(vals[21])
    weights = stocGradAscent1(array(data), labels, 500)
    err,numtest = 0,0
    for line in open("horseColicTest.txt"):
        numtest+=1
        vals = list(map(float,line.split()))
        if int(classifyVector(array(vals[:21]),weights)) != int(vals[21]):
            err+=1
    errRate = err/numtest
    print(f"Error rate: {errRate}")
    return errRate

def multiTest():
    n,err = 10,0.0
    for _ in range(n):
        err += colicTest()
    print(f"Average error rate: {err/n}")
