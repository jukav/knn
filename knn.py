#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Aplications for data analysis
# 1. Exersice

# Code is modified from website: 
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/


# calculate distance
import math # load math module
def euclideanDistance(instance1, instance2, length): #define function
    distance = 0
    for x in range(length): # calculate distances
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance) # return value


# In[2]:


# get neighbors from training set
import operator # import module
def getNeighbors(trainingSet, testInstance, k): # define function
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)): 
        dist = euclideanDistance(testInstance, trainingSet[x], length) # call euclidianDistance function
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1)) # sort distances
    neighbors = []
    for x in range(k): # get neighbors
        neighbors.append(distances[x][0])
    return neighbors # return neighbors


# In[3]:


import operator
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# In[4]:


# calculate prediction accuracy
def getAccuracy(testSet, predictions): # define function
    correct = 0
    for x in range(len(testSet)): # calculate accuracy
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0 # return prediction accuracy percent


# In[5]:


# read iris- file
import csv
with open('iris.data.txt', 'r') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print (', '.join(row))


# In[6]:


# split function to devide data to train and test set
import csv
import random
def loadDataset(filename, split, trainingSet=[] , testSet=[]): # define function
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split: # random split to train and test set
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# In[7]:


# load iris data
trainingSet=[]
testSet=[]
loadDataset('iris.data.txt', 1, trainingSet, testSet)
print ("Train: " + repr(len(trainingSet)))
print ("Test: " + repr(len(testSet)))
dataSet=trainingSet


# In[8]:


# cross validation for dataset
from random import seed # import needed modules
from random import randrange
 
# Split a dataset into k folds
def cross_validation_split(dataset, folds=3): # define fuction
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds) # calculate fold size
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size: # create folds
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split # return folds
 
# test cross validation split
seed(1)
folds = cross_validation_split(dataSet,10) # call function
#print(folds)


# In[9]:


# calculate test error for predictions with crossvalidation
maccur=0
l=0

for i in range (len(folds)):
    trainingSet=dataSet
    testSet=folds[i]
    
    predictions=[]
    k = 10
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    l=l+1
    maccur=maccur+accuracy
    print('Prediction accuracy: ' + repr(accuracy) + '%')
print('Prediction mean accuracy procent:')
print(maccur/l)


# In[10]:


# Prediction accuracy with cross validation (10 fold) wih different k-values
#
# k=3    accuracy: 95,71 %
# k=4    accuracy: 97,14 %
# k=5    accuracy: 97,14 %
# k=6    accuracy: 97,86 %
# k=7    accuracy: 97,86 %
# k=8    accuracy: 97,86 %
# k=9    accucacy: 97,14 %"



# In[ ]:





# In[ ]:




