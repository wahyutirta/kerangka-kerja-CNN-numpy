# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:23:01 2021

@author: ASUS
"""

import numpy as np
import cv2
import os
from tqdm import tqdm
from einops import rearrange, reduce, repeat

class Data:
    """
        Dokumentasi
        
        input : 
        output :
        
    """
    def __init__(self, workPath, imagePath):
        self.dataPath = os.path.join(workPath[0],imagePath) #image path
        self.imagePath = imagePath


    def unison_shuffled_copies_4(self,a , b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def load(self,trainRatio=0.8,testRatio=0.2):
        arr_img = []
        arr_label = []
        arr_Namelabel = []
        self.count = 0
        for i, (dirpath, dirnames, filenames) in tqdm(enumerate(os.walk(self.imagePath)), desc= "Loading Image Data"):
            #print('{} {} {}'.format(repr(dirpath), repr(dirnames), repr(filenames)))
            #print(i)
            if dirpath is not self.imagePath:
                dirpath_components = dirpath.split("/")
                arr_img = []
                arr_label = []
                semantic_label = dirpath_components[-1]
                
                _, label = os.path.split(semantic_label)

                #print("\nProcessing {}, {}".format(semantic_label,i))
                arr_Namelabel.append(label)
                self.count = 0

                for f in filenames:
                    #load images
                    file_path = os.path.join(dirpath, f)
                    #print(file_path)
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = rearrange(img, ' h w c ->  c h w ')
                    #print("---",img.shape,file_path)
                    a,b,c = img.shape
                    if b != 64 or c != 64:
                        print("---",img.shape, file_path)
                    arr_img.append(img)
                    arr_label.append(i-1)
                    self.count+=1
                    
                
                

mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
workPath = os.path.split(mainPath) #path working folder (whole file project)
imagePath = "data_jepun"
data = Data(workPath,imagePath)
data.load(trainRatio=0.8,testRatio=0.2)