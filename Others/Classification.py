### TRAINING AND TESTING SET DISTRIBUTION IS IMBALANCED SO USE DIV1.PY INSTEAD. NOW USING SHUFFLING


import sys
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import shuffle
import os
import numpy
import itertools
import cPickle as pickle
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
#np.set_printoptions(threshold=np.nan)

with open('hog_all1','rb') as f:
	hist=pickle.load(f)

with open('Hog_labels_all1','rb') as f1:
	lb=pickle.load(f1)

list1_shuf = []
list2_shuf = []
index_shuf = range(len(lb))
# shuffle(index_shuf)
for i in index_shuf:
    list1_shuf.append(hist[i])
    list2_shuf.append(lb[i])


train=list1_shuf[:int(0.8*len(lb))]
train_lb=list2_shuf[:int(0.8*len(lb))]

test=list1_shuf[int(0.8*len(lb)):]
test_lb=list2_shuf[int(0.8*len(lb)):]

clfSVM = svm.LinearSVC(loss='squared_hinge')
clfSVM.fit(train, train_lb)
predSVM=clfSVM.predict(test)

clfRAN=ensemble.RandomForestClassifier(n_estimators=40)
clfRAN=clfRAN.fit(train, train_lb)
predRAN=clfRAN.predict(test)


clfDEC=tree.DecisionTreeClassifier()
clfDEC=clfDEC.fit(train, train_lb)
predDEC=clfDEC.predict(test)


total,correlation = 0.0,0.0
for label in test_lb:
    if predDEC[total] == label:
        correlation += 1
    total += 1
print "Accuracy of decision tree is", correlation / total


total,correlation = 0.0,0.0
for label in test_lb:
    if predSVM[total] == label:
        correlation += 1
    total += 1
print "Accuracy of svm is", correlation / total

total,correlation = 0.0,0.0
for label in test_lb:
    if predRAN[total] == label:
        correlation += 1
    total += 1
print "Accuracy of random forest is", correlation / total

