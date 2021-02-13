from numpy import *


def loadSimpData():
    data = matrix([[1. ,2.1],
        [2. , 1.1],
        [1.3 , 1.],
        [1. , 1.],
        [2. , 1.]])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data, labels

def loadDataSet(fn):
    with open(fn) as f:
        numFeat = len(f.readline().strip().split('\t'))
        data,labels = [],[]
        f.seek(0)
        for line in f:
            vals = list(map(float,line.strip().split('\t')))
            data.append(vals[:-2])
            labels.append(vals[-1])
    return data,labels


def stumpClassify(data,dimen,threshVal,threshIneq):
    ret = ones((shape(data)[0],1))
    if threshIneq == 'lt':
        ret[data[:,dimen] <= threshVal] = -1.0
    else:
        ret[data[:,dimen] > threshVal] = -1.0
    return ret

def buildStump(data,labels,D):
    data,labels = mat(data),mat(labels).T
    m,n = shape(data)
    numSteps,bestStump,bestClasEst,minErr = 10.0,{},mat(zeros((m,1))),inf
    for i in range(n):
        rangeMin,rangeMax = data[:,i].min(),data[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predVals = stumpClassify(data, i, threshVal, inequal)
                err = mat(ones((m,1)))
                err[predVals==labels] = 0
                wErr = D.T*err
                print(f"split dim {i}, thresh {threshVal:.3f}, {inequal}, {wErr[0,0]:.3f}")
                if wErr < minErr:
                    minErr = wErr
                    bestClasEst = predVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minErr,bestClasEst


def adaBoostTrainDS(data,labels,numIt=40):
    weak = []
    m = shape(data)[0]
    D = mat(ones((m,1))/m)
    aggEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,est = buildStump(data,labels,D)
        print("D: ",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weak.append(bestStump)
        print("classEst: ", est.T)
        expon = multiply(-1*alpha*mat(labels).T,est)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggEst += alpha*est
        print("aggClassEst: ", aggEst.T)
        aggErr = multiply(sign(aggEst) != mat(labels).T, ones((m,1)))
        errRate = aggErr.sum()/m
        print("total error: ", errRate)
        if errRate == 0.0:
            break
    return weak,aggEst


def adaClassify(data,classifiers):
    data = mat(data)
    m = shape(data)[0]
    aggEst = mat(zeros((m,1)))
    for classifier in classifiers:
        est = stumpClassify(data, classifier['dim'],\
                                classifier['thresh'],\
                                classifier['ineq'])
        aggEst += classifier['alpha']*est
        print(aggEst)
    return sign(aggEst)


def plotROC(predStr, labels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0.0
    numPos = sum(array(labels)==1.0)
    yk,xk = 1/numPos,1/(len(labels)-numPos)
    print(xk,yk)
    indices = predStr.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for i in indices.tolist()[0]:
        if labels[i] == 1.0:
            dx,dy = 0,yk
        else:
            dx,dy = xk,0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-dx],[cur[1],cur[1]-dy],c='b')
        cur = (cur[0]-dx,cur[1]-dy)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for AdaBoosst Horse Colic Detection System")
    ax.axis([0,1,0,1])
    plt.show()
    print("Area under curve: ",ySum*xk)
