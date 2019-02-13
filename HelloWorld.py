from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

import numpy as np

# CHALLENGE - create 3 more classifiers...
# 1 
# 2
# 3

clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)


# CHALLENGE compare their results and print the best one!

pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y,pred_tree)*100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y,pred_svm)*100
print('Accuracy for SVC: {}'.format(acc_svm))

pred_percep = clf_perceptron.predict(X)
acc_percep = accuracy_score(Y,pred_percep)*100
print('Accuracy for Perceptron: {}'.format(acc_percep))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y,pred_KNN)*100
print('Accuracy for KNeighborsClassifier: {}'.format(acc_KNN))


index = np.argmax([acc_svm,acc_percep,acc_KNN])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))


