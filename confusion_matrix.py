# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:34:14 2021

@author: ASUS
"""

import numpy as np

import seaborn as sns
import matplotlib as plt
from lenet5 import *
from Databunga import Data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def convertName(array, label):
    tempArray = []
    for value in array:
        tempArray.append(label[value])
    return tempArray


mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
workPath = os.path.split(mainPath) #path working folder (whole file project)
imagePath = "data_jepun"
    
data = Data(workPath, imagePath)
X_train, trainLabel, fNameTrain ,X_test, testLabel, fNameTest = data.load()
kelas = data.jum_kelas
len_label = trainLabel.shape[0]
    
Y_train = np.zeros((len_label,kelas))
Y_train[np.arange(len_label), trainLabel[range(0, len_label)]] = 1
    
kelas = data.jum_kelas
len_label = testLabel.shape[0]
y_true = np.zeros((len_label, kelas))
y_true[np.arange(len_label), testLabel[range(0, len_label)]] = 1
   
label = data.labelName
    
method = "adam"
epochs = 201
batch = 32
learningRate = 0.0001

mylenet = LENET5(X_train, Y_train, X_test, y_true, method=method,epochs=epochs, batch=batch, learningRate=learningRate )
imgpath= "C:/Users/ASUS/Documents/py/cnn-numpy/data_jepun/Plumeria_rubra_L_tri_color/sudamala_(1).jpg"
temp = os.path.split(imgpath)
        
""" load training history """
mylenet.load_train_details(mainPath=mainPath,epochs=epochs,method=method, batch=batch, learningRate=learningRate )
    
#""" testing one image """
#print("Params: batch=", batch, " learning rate=", learningRate, "method=", method, "epochs=", epochs)
        
mylenet.load_parameters(mainPath=mainPath,epochs=epochs,method=method, batch=batch, learningRate=learningRate)
    
y_true, y_pred = mylenet.lenet_predictions_return(mylenet, mylenet.layers,X_test, y_true,fNameTest, data.labelName)
prob = mylenet.one_image(mylenet.layers, imgpath )
print("\nFile Name ::", temp[1], " Tipe bunga ::", data.labelName[np.argmax(prob)], "||" ,
          "confidence ::", prob[0,np.argmax(prob)])


#y_true = convertName(y_true, label)
#y_pred = convertName(y_pred, label)
fOne = f1_score(y_true, y_pred, average = None)
print(fOne)
precision_score = precision_score(y_true, y_pred, average = None)
recall_score = recall_score(y_true, y_pred, average = None)


cf_matrix = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
print("tn, fp, fn, tp", tn, fp, fn, tp)
#sns.heatmap(cf_matrix, cmap ='coolwarm', annot=True, linewidth=1, fmt="d")

confusion = confusion_matrix(y_true, y_pred)
print('Confusion Matrix\n')
print(confusion)

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_true, y_pred, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_true, y_pred, target_names=['Bali', 'Cendana', 'Indonesia pink', 'Tri color Sudamala']))
