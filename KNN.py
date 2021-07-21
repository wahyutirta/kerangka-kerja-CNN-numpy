# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:37:19 2021

@author: ASUS
"""
import timeit
from sklearn.metrics import accuracy_score
from sklearn import metrics
from tqdm import tqdm
from Databunga import Data
import os
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors

mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
workPath = os.path.split(mainPath) #path working folder (whole file project)
imagePath = "data_jepun"
    
data = Data(workPath, imagePath)

"""
X, y_train, xTest, y_test = data.loadGlcm()

clf = neighbors.KNeighborsClassifier()
clf.fit(X, y_train)
print(clf)
y_pred = clf.predict(xTest)
#print (metrics.classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Bali', 'Cendana', 'Indonesia pink', 'Tri color Sudamala']))
"""

X, y_train, xTest, y_test = data.loadHistogram()

clf = neighbors.KNeighborsClassifier()
clf.fit(X, y_train)
print(clf)
y_pred = clf.predict(xTest)
#print (metrics.classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Bali', 'Cendana', 'Indonesia pink', 'Tri color Sudamala']))