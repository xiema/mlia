from numpy import *

def loadDataSet(fn):
    data = []
    with open(fn) as f:
        for line in f:
            data.append(list(map(float,line.strip().split('\t'))))
    return data


def binSplitDataSet(data, ft, val):
    #print(data[nonzero(data[:,ft]<=val)[0],:])
    #data0,data1 = nonzero(data[:,ft]>val),nonzero(data[:,ft]<=val)
    m0 = data[nonzero(data[:,ft]>val)[0],:]
    m1 = data[nonzero(data[:,ft]<=val)[0],:]
    return m0 ,m1

def regLeaf(data):
    return mean(data[:,-1])

def regErr(data):
    return var(data[:,-1]) * shape(data)[0]

def createTree(data, leafType=regLeaf, errType=regErr, ops=(1,4)):
    ft,val = chooseBestSplit(data,leafType,errType,ops)
    if ft==None: return val
    ldata,rdata = binSplitDataSet(data, ft, val)
    ltree = createTree(ldata,leafType,errType,ops)
    rtree = createTree(rdata,leafType,errType,ops)
    return {'spInd': ft, 'spVal': val, 'left': ltree, 'right': rtree}


def chooseBestSplit(data, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS,tolN = ops[0],ops[1]
    if len(set(data[:,-1].T.tolist()[0])) == 1:
        return None, leafType(data)
    m,n = shape(data)
    S = errType(data)
    bestS,bestFt,bestVal = inf,0,0
    for ft in range(n-1):
        for val in set(data[:,ft].T.tolist()[0]):
            m0,m1 = binSplitDataSet(data, ft, val)
            if shape(m0)[0] < tolN or shape(m1)[0] < tolN:
                continue
            newS = errType(m0) + errType(m1)
            if newS < bestS:
                bestS,bestFt,bestVal=newS,ft,val
    if S-bestS < tolS:
        return None, leafType(data)
    m0,m1 = binSplitDataSet(data, bestFt, bestVal)
    if shape(m0)[0]<tolN or shape(m1)[0]<tolN:
        return None, leafType(data)
    return bestFt,bestVal


def isTree(obj):
    return type(obj).__name__=='dict'

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['right']+tree['left'])/2.0

def prune(tree, testData):
    if shape(testData)[0]==0: return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        l,r = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'],l)
    if isTree(tree['right']): tree['right'] = prune(tree['right'],r)
    if not isTree(tree['left']) and not isTree(tree['right']):
        l,r = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(l[:,-1]-tree['left'],2)) +\
                    sum(power(r[:,-1]-tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
    return tree


def linearSolve(dataSet):
    m,n = shape(dataSet)
    X,Y = mat(ones((m,n))),mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError("Can't invert singular matrix. Try increasing ops[1]")
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = zeros((m,1))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
