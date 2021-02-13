from numpy import *

def loadDataSet(fn):
    with open(fn) as f:
        n = len(f.readline().strip().split('\t'))-1
        data,labels = [],[]
        f.seek(0)
        for line in f:
            vals = list(map(float, line.strip().split('\t')))
            data.append(vals[:-1])
            labels.append(vals[-1])
    return data,labels


def standRegres(x, y):
    x,y = mat(x),mat(y).T
    xTx = x.T*x
    if linalg.det(xTx) == 0.0:
        print("Matrix is singular. Can't invert.")
        return
    w = xTx.I * (x.T*y)
    return w





def lwlr(testPoint, x,y,k=1.0):
    x,y = mat(x),mat(y).T
    m = shape(x)[0]
    w = mat(eye((m)))
    for i in range(m):
        diff = testPoint - x[i,:]
        w[i,i] = exp(diff*diff.T/(-2.0*k**2))
    xTx = x.T * (w*x)
    if linalg.det(xTx) == 0.0:
        print("Matrix is singular. Can't invert.")
        return
    w = xTx.I * (x.T * (w*y))
    return testPoint * w

def lwlrTest(testArr,x,y,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],x,y,k)
    return yHat



def ridgeRegres(x, y, lam=0.2):
    xTx = x.T*x
    denom = xTx + eye(shape(x)[1])*lam
    if linalg.det(denom) == 0.0:
        print("Can't invert singular matrix")
        return
        ws = denom.I * (x.T*y)
        return ws

def ridgeTest(x,y):
    x,y = mat(x),mat(y).T
    yMean = mean(y,0)
    y = y - yMean
    xMean = mean(x,0)
    xVar = var(x,0)
    x = (x-xMean)/xVar
    n = 30
    w = zeros((n,shape(x)[1]))
    for i in range(n):
        w[i,:] = ridgeRegres(x,y,exp(i-10)).T
        return w

def stageWise(x, y, eps=0.01, numIt=100):
    x,y = mat(x),mat(y).T
    y = y - mean(y,0)
    x = (x-mean(x,0))/var(x,0)
    m,n = shape(x)
    w = zeros((n,1))
    wTest,wMax = w.copy(),w.copy()
    returnMat = zeros((numIt,n))
    for i in range(numIt):
        print(w.T)
        lerr = inf;
        for j in range(n):
            for sign in [-1,1]:
                wTest = w.copy()
                wTest[j] += eps*sign
                yTest = x*wTest
                rssE = rssError(y.A,yTest.A)
                if rssE < lerr:
                    lerr = rssE
                    wMax = wTest
        w = wMax.copy()
        returnMat[i,:]=w.T
    return returnMat


def rssError(y,yHat):
    return ((y-yHat)**2).sum()
