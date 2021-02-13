from numpy import *

def loadDataSet(fn):
    data,labels = [],[]
    with open(fn) as f:
        for line in f:
            v = list(map(float,line.split()))
            data.append(v[:2])
            labels.append(v[2])
    return data,labels

def img2vector(filename):
    r = zeros((1,1024))
    with open(filename, 'r') as file:
        for i in range(32):
            line = file.readline()
            for j in range(32):
                r[0,32*i+j] = int(line[j])
    return r

def loadImages(dirname):
    from os import scandir
    dir = list(scandir(dirname))
    data,labels = zeros((len(dir),1024)),[]
    for i,fe in enumerate(dir):
        c = int(fe.name.split('.')[0].split('_')[0])
        labels.append(-1 if c==9 else 1)
        data[i,:] = img2vector(fe.path)
    return data,labels



def selectJrand(i,m):
    j=i
    while j==i:
        j = int(random.uniform(0,m))
    return j

def clamp(x,lo,hi):
    return min(max(x,lo),hi)


def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin':
        K = X * A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K / (-1*kTup[1]**2))
    else:
        raise NameError("Unknown kernel")
    return K


class optStruct:
    def __init__(self, data, labels, C, tol, kTup):
        self.X = data
        self.labels = labels
        self.C = C
        self.tol = tol
        self.m = shape(data)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labels).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labels[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK,maxDeltaE,Ej = -1,0,0
    oS.eCache[i] = [1,Ei]
    validCache = nonzero(oS.eCache[:,0].A)[0]
    if len(validCache) > 1:
        for k in validCache:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek)
            if deltaE > maxDeltaE:
                maxK,maxDeltaE,Ej = k,deltaE,Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labels[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C) or\
            (oS.labels[i]*Ei > oS.tol) and (oS.alphas[i] > 0):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold,alphaJold = oS.alphas[i].copy(),oS.alphas[j].copy()
        if oS.labels[i] != oS.labels[j]:
            L = max(0, oS.alphas[j]-oS.alphas[i])
            H = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labels[j]*(Ei-Ej)/eta
        oS.alphas[j] = clamp(oS.alphas[j],L,H)
        updateEk(oS,j)
        if abs(oS.alphas[j]-alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labels[j]*oS.labels[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        b1 = oS.b - Ei - oS.labels[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] \
            - oS.labels[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labels[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] \
            - oS.labels[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if 0 < oS.alphas[i] and oS.C > oS.alphas[i]:
            oS.b = b1
        elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]:
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0

def smoP(data, labels, C, tol, maxIter, kTup=('lin',0)):
    oS = optStruct(mat(data),mat(labels).transpose(),C,tol,kTup)
    iter = 0
    entireSet,changed = True,0
    while iter<maxIter and (changed > 0 or entireSet):
        changed = 0
        if entireSet:
            for i in range(oS.m):
                changed += innerL(i,oS)
            print(f"fullSet, iter:{iter} i:{i} changed:{changed}")
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                changed += innerL(i,oS)
                print(f"non-bound, iter:{iter} i:{i} changed:{changed}")
                iter += 1
        if entireSet:
            entireSet = False
        elif changed == 0:
            entireSet = True
        print(f"iteration num: {iter}")
    return oS.b,oS.alphas


def smoSimple(data, labels, C, toler, maxIter):
    data,labels = mat(data),mat(labels).transpose()
    m,n = shape(data)
    b,iter,alphas = 0.0,0,mat(zeros((m,1)))
    while iter < maxIter:
        changed = 0
        for i in range(m):
            fXi = float(multiply(alphas,labels).T * (data*data[i,:].T)) + b
            Ei = fXi - float(labels[i])
            if (labels[i]*Ei < -toler and alphas[i] < C) or \
                    (labels[i]*Ei > toler and alphas[i] > 0):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labels).T * (data*data[j,:].T)) + b
                Ej = fXj - float(labels[j])
                alphaIold,alphaJold = alphas[i].copy(),alphas[j].copy()
                if labels[i] != labels[j]:
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * data[i,:]*data[j,:].T - data[i,:]*data[i,:].T - data[j,:]*data[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labels[j] * (Ei-Ej)/eta
                alphas[j] = clamp(alphas[j],L,H)
                if abs(alphas[j]-alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labels[j]*labels[i]*(alphaJold-alphas[j])
                b1 = b - Ei - labels[i]*(alphas[i]-alphaIold)*data[i,:]*data[i,:].T \
                    - labels[j]*(alphas[j]-alphaJold)*data[i,:]*data[j,:].T
                b2 = b - Ej - labels[i]*(alphas[i]-alphaIold)*data[i,:]*data[j,:].T \
                    - labels[j]*(alphas[j]-alphaJold)*data[j,:]*data[j,:].T
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1+b2)/2.0
                changed+=1
                print(f"iter:{iter} i:{i}, changed {changed}")
        if changed == 0:
            iter += 1
        else:
            iter = 0
        print(f"iteration number: {iter}")
    return b,alphas


def testRbf(k1=1.3):
    data,labels = loadDataSet("testSetRBF.txt")
    b,alphas = smoP(data, labels, 200, 0.0001, 10000, ('rbf', k1))
    data,labels = mat(data), mat(labels).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs,labelSV = data[svInd],labels[svInd]
    print(f"{shape(sVs)[0]} Support Vectors")
    m,n = shape(data)
    err = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,data[i,:],('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) +b
        if sign(predict) != sign(labels[i]):
            err += 1
    print(f"Training error rate: {float(err)/m}")

    data,labels = loadDataSet("testSetRBF2.txt")
    err = 0
    data,labels = mat(data),mat(labels).transpose()
    m,n = shape(data)
    for i in range(m):
        kernelEval = kernelTrans(sVs,data[i,:],('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labels[i]):
            err += 1
    print(f"Test error rate: {float(err)/m}")


def testDigits(kTup=('rbf',10)):
    data,labels = loadImages('trainingDigits')
    b,alphas = smoP(data, labels, 200, 0.0001, 10000, kTup)
    data,labels = mat(data),mat(labels).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = data[svInd]
    labelSV = labels[svInd]
    print(f"{shape(sVs)[0]} Support Vectors")
    m,n = shape(data)
    err = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,data[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labels[i]):
            err += 1
    print(f"Training error rate: {float(err)/m}")
    data,labels = loadImages('testDigits')
    err = 0
    data,labels = mat(data),mat(labels).transpose()
    m,n = shape(data)
    for i in range(m):
        kernelEval = kernelTrans(sVs,data[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labels[i]):
            err += 1
    print(f"Test error rate: {float(err)/m}")


def calcWs(alphas,data,labels):
    X,labels = mat(data),mat(labels).transpose()
    m,n = shape(X)
    w =  zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labels[i],X[i,:].T)
    return w
