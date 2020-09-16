from numpy import *

def loadDataSet(fn):
    dataMat = []
    with open(fn) as f:
        for line in f:
            vals = line.strip().split('\t')
            dataMat.append(list(map(float,vals)))
    return dataMat

def distEuclid(matA,vecB):
    return sum(power(matA-vecB,2),axis=1)

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    # centroids = mat(zeros((k,n)))
    # for j in range(n):
    #     minJ = min(dataSet[:,j])
    #     rangeJ = float(max(dataSet[:,j]) - minJ)
    #     centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    minJ,maxJ = amin(dataSet,axis=0),amax(dataSet,axis=0)
    centroids = (maxJ-minJ) * random.rand(k,n) + minJ
    return centroids

def kMeans(dataSet, k, distMeas=distEuclid, createCent=randCent):
    m = shape(dataSet)[0]
    centroids = createCent(dataSet,k)
    assignments = tile([0,inf],(m,1))
    changed = True
    while changed:
        changed = False
        for i in range(k):
            distances = distMeas(dataSet,centroids[i])
            changedIndices = nonzero(distances < assignments[:,1])[0]
            if changedIndices.any():
                assignments[changedIndices,0] = i
                assignments[changedIndices,1] = distances[changedIndices]
                changed = True
        #print(centroids)
        for i in range(k):
            pts = dataSet[nonzero(i==assignments[:,0])[0]]
            centroids[i,:] = mean(pts, axis=0)
    return centroids, assignments

def biKMeans(dataSet, k, distMeas=distEuclid):
    m = shape(dataSet)[0]
    assignments = zeros((m,2))
    cent0 = mean(dataSet, axis=0).tolist()[0]
    centList = [cent0]
    assignments[:,1] = distMeas(dataSet,cent0)
    while len(centList) < k:
        lo = inf
        for i in range(len(centList)):
            pts = dataSet[assignments[:,0]==i,:]
            centSub,splitAssignments = kMeans(pts, 2, distMeas)
            sseSplit = sum(splitAssignments[:,1])
            sseNotSplit = sum(assignments[assignments[:,0]!=i,1])
            print(f"SSE Split|Not Split: {sseSplit} | {sseNotSplit}")
            if sseSplit + sseNotSplit < lo:
                bestCent,bestAssignments = i,splitAssignments
                bestNewCents = centSub
                lo = sseSplit + sseNotSplit
        bestAssignments[:,0] = bestAssignments[:,0]*len(centList) + (1-bestAssignments[:,0])*bestCent
        print(f"bestCent: {bestCent}")
        print(f"len(bestAssignments): {len(bestAssignments)}")
        centList[bestCent] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        assignments[assignments[:,0]==bestCent,:] = bestAssignments

    return centList, assignments
