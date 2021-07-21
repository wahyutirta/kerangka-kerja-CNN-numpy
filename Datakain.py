import numpy as np
import cv2
import os
from tqdm import tqdm
from einops import rearrange, reduce, repeat

class Data:
    def __init__(self, workPath, imagePath):
        self.dataPath = os.path.join(workPath[0],imagePath) #image path
        self.imagePath = imagePath

    
    def unison_shuffled_copies_4(self,a , b):
        """
        DOKUMENTASI
        
        """
    
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def load(self,trainRatio=0.8,testRatio=0.2):
        """
        DOKUMENTASI 
        
        """
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

                for f in filenames:
                    #load images
                    file_path = os.path.join(dirpath, f)
                    #print(file_path)
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = rearrange(img, ' h w c ->  c h w ')
                    arr_img.append(img)
                    arr_label.append(i-1)
                    self.count+=1
                    
                    #print(arr_img.shape)
                    
                dataset = np.array(arr_img)
                label = np.array(arr_label)
                dataset, label = self.unison_shuffled_copies_4(dataset, label)
                #print(i)
                if(i == 1):
                    self.data0 = np.array(dataset, dtype='float64') / 255
                    self.label0 = np.array(label)
                elif(i == 2):
                    self.data1 = np.array(dataset, dtype='float64') / 255
                    self.label1 = np.array(label)
                elif(i == 3):
                    self.data2 = np.array(dataset, dtype='float64') / 255
                    self.label2 = np.array(label)
                elif(i == 4):
                    self.data3 = np.array(dataset, dtype='float64') / 255
                    self.label3 = np.array(label)
        
        self.labelName = np.array(arr_Namelabel)
        #print(self.labelName)
        self.jum_kelas = len(self.labelName)
        n_train = (int(self.count * trainRatio) // self.jum_kelas) 
        n_test = (int(self.count * testRatio ) // self.jum_kelas) - 1
       
        self.trainSet = np.concatenate((self.data0[0:n_train,:,:,:],
                                       self.data1[0:n_train,:,:,:],
                                       self.data2[0:n_train,:,:,:],
                                       self.data3[0:n_train,:,:,:],), axis = 0)
        self.trainLabel = np.concatenate((self.label0[0:n_train,],
                                       self.label1[0:n_train,],
                                       self.label2[0:n_train,],
                                       self.label3[0:n_train,],), axis = 0)

        self.testSet = np.concatenate((self.data0[n_train:,:,:,:],
                                       self.data1[n_train:,:,:,:],
                                       self.data2[n_train:,:,:,:],
                                       self.data3[n_train:,:,:,:],), axis = 0)
        self.testLabel = np.concatenate((self.label0[n_train:,],
                                       self.label1[n_train:,],
                                       self.label2[n_train:,],
                                       self.label3[n_train:,],), axis = 0)

        self.trainSet, self.trainLabel = self.unison_shuffled_copies_4(self.trainSet, self.trainLabel)
        self.testSet, self.testLabel  = self.unison_shuffled_copies_4(self.testSet, self.testLabel)
        
        del self.data0, self.data1, self.data2, self.data3, \
            self.label0, self.label1, self.label2, self.label3
            
        return self.trainSet, self.trainLabel, self.testSet, self.testLabel

#mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
#workPath = os.path.split(mainPath) #path working folder (whole file project)
#imagePath = "data_kain"
#data = Data(workPath,imagePath)
#data.load(trainRatio=0.8,testRatio=0.2)