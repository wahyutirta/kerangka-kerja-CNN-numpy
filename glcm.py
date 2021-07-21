# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 22:08:52 2021

@author: User
"""

import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from skimage.feature import greycomatrix, greycoprops

DATASET_PATH = "data_jepun"
JSON_PATH= "data.json"
angless= [0., np.pi/4., np.pi/2., 3.*np.pi/4.]

#angless= [0.]

features = ['dissimilarity']

#features = ['dissimilarity','energy','homogeneity','contrast']
# 0., np.pi/4., np.pi/2., 3.*np.pi/4.


def save_glcm(dataset_path, json_path):
    # dict penyimpanan data
    data = {
        "mapping" : [],
        "glcm": [],
        "labels":[]
        }
    
    for i, (dirpath, disnames, filenames) in enumerate(os.walk(dataset_path)):
        #i = kelas
        label_array = []
        data_array = []
        temp_arr = []
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split("/") 
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            #print("\nProcessing {}".format(semantic_label))
            counter = 0
            
            #process glcm
            temp_list = []
            for f in tqdm(filenames, desc="Processing {}\t {}".format(semantic_label, counter), position=0):
                #save img class
                counter +=1
                #data["labels"].append(i-1)
                label_array.append(i-1)
                
                #load image
                file_path = os.path.join(dirpath, f)
                
                gray = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                
                glcm = greycomatrix(gray, distances=[5], angles=angless, levels=256, symmetric=True, normed=True)                
                
                #temp_list = np.empty(shape=[1,0]) # empty = [ [  ] ]
                for feature in features:
                    
                    temp = greycoprops(glcm, feature)
                    temp_arr.append(temp[0].tolist())
                    
    label_array = np.array(label_array)
    return temp_arr, label_array
                         

data, label = save_glcm(DATASET_PATH, JSON_PATH)
data = np.array(data)