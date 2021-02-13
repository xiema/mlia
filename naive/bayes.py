import re
from random import sample
from numpy import *

def createVocabList(dataSet):
    vocabSet = set()
    for d in dataSet:
        vocabSet |= set(d)
    return list(vocabSet)

def set2Vec(vocabList, inputSet):
    vec = [0 for _ in vocabList]
    for word in inputSet:
        if word in vocabList:
            vec[vocabList.index(word)] = 1
        else:
            print(f'Unknown word: {word}')
    return vec

def trainNB0(trainMatrix, trainCategory):
    nDoc, nWord = len(trainMatrix), len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(nDoc)
    p0Num, p1Num = zeros(nWord), zeros(nWord)
    p0Den, p1Den = 0.0, 0.0
    for i in range(nDoc):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Den += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Den += sum(trainMatrix[i])
    p1V = p1Num/p1Den
    p0V = p0Num/p0Den
    return p0V, p1V, pAbusive

def classifyNB(v, p0V, p1V, pClass1):
    p1 = sum(v*p1V) + log(pClass1)
    p0 = sum(v*p0V) + log(1.0 - pClass1)
    return 1 if p1>p0 else 0


pat = re.compile(r'\W+')
def textParse(bigString):
    tokens = pat.split(bigString)
    return [t.lower() for t in tokens if len(t) > 2]

def spamTest():
    docList=[] ; classList=[] ; fullText=[]
    for i in range(1,26):
        with open(f'naive/email/spam/{i}.txt', encoding='latin-1') as f:
            wordList = textParse(f.read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        with open(f'naive/email/ham/{i}.txt', encoding='latin-1') as f:
            wordList = textParse(f.read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = set(list(range(50)))
    testSet = set(sample(trainingSet, 10))
    trainingSet -= testSet
    trainMat,trainClasses = [],[]
    for i in trainingSet:
        trainMat.append(set2Vec(vocabList, docList[i]))
        trainClasses.append(classList[i])
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for i in testSet:
        v = set2Vec(vocabList, docList[i])
        if classifyNB(array(v), p0V, p1V, pSpam) != classList[i]:
            errorCount += 1
    errorRate = errorCount/len(testSet)
    print(f'Error rate: {errorRate}')

if __name__ == '__main__':
    spamTest()
