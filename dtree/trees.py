from math import log
from collections import Counter
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = Counter()
    for featVec in dataSet:
        labelCounts[featVec[-1]] += 1
    shannonEnt = 0.0
    for v in labelCounts.values():
        p = float(v)/numEntries
        shannonEnt -= p * log(p,2)
    return shannonEnt

def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    ret = []
    for featVec in dataSet:
        if featVec[axis] == value:
            ret.append(featVec[:axis]+featVec[axis+1:])
    return ret

def chooseSplitFeature(dataSet):
    featCount = len(dataSet[0])-1
    base = calcShannonEnt(dataSet)
    bestGain, bestFeat = 0.0, -1
    for i in range(featCount):
        gain = base
        for v in set(d[i] for d in dataSet):
            subDataSet = splitDataSet(dataSet, i, v)
            p = len(subDataSet)/float(len(dataSet))
            gain -= p * calcShannonEnt(subDataSet)
        if gain > bestGain:
            bestGain, bestFeat = gain, i
    return bestFeat

def majorityCnt(classList):
    cnt = Counter(classList)
    return cnt.most_common(1)[0]

def createTree(dataSet, labels):
    classList = [d[-1] for d in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseSplitFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    t = {bestFeatLabel:{}}
    for v in set(d[bestFeat] for d in dataSet):
        t[bestFeatLabel][v] = createTree(splitDataSet(dataSet,bestFeat,v),labels[:bestFeat]+labels[bestFeat+1:])
    return t

def classify(inTree, featLabels, testVec):
    featLabel, featDict = [(k,v) for k,v in inTree.items()][0]
    featIdx = featLabels.index(featLabel)
    for k,v in featDict.items():
        if testVec[featIdx] == k:
            if type(v).__name__ == 'dict':
                return classify(v, featLabels, testVec)
            else:
                return v

def storeTree(inTree, fn):
    import pickle
    with open(fn, 'w') as f:
        pickle.dump(inTree,f)
def grabTree(fn):
    import pickle
    with open(fn) as f:
        return pickle.load(f)
