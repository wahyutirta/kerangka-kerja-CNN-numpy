import numpy as np
import cv2
import os
import math
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from skimage.feature import greycomatrix, greycoprops

class Data:
    def __init__(self, workPath, imagePath):
        self.dataPath = os.path.join(workPath[0],imagePath) #image path
        self.imagePath = imagePath
    """
        DOKUMENTASI 
        
    """    
    @staticmethod
    def unison_shuffled_copies_4( a , b, c):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p] , c[p]
    
    @staticmethod
    def unison_shuffled_copies_2( a , b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def loadLabel(self):
        """
        DOKUMENTASI 
        
        """
        arr_Namelabel = []
        self.count = 0
        for i, (dirpath, dirnames, filenames) in tqdm(enumerate(os.walk(self.imagePath)), desc= "Loading Image Data"):
            #print('{} {} {}'.format(repr(dirpath), repr(dirnames), repr(filenames)))
            #print(i)
            if dirpath is not self.imagePath:
                dirpath_components = dirpath.split("/")
                listImageTrain = []
                listLabelTrain = []
                listImageTest = []
                listLabelTest = []
                testFName = []
                trainFName = []
                semantic_label = dirpath_components[-1]
                
                _, label = os.path.split(semantic_label)

                #print("\nProcessing {}, {}".format(semantic_label,i))
                arr_Namelabel.append(label)
        labelArray = np.array(arr_Namelabel)
        return labelArray
    
    def load(self,trainRatio=0.8,testRatio=0.2):
        
        """
        DOKUMENTASI 
        
        """
        
        temp_mod = math.ceil(trainRatio/testRatio)
        #arr_img = []
        #arr_label = []
        arr_Namelabel = []
        self.count = 0
        for i, (dirpath, dirnames, filenames) in tqdm(enumerate(os.walk(self.imagePath)), desc= "Loading Image Data"):
            #print('{} {} {}'.format(repr(dirpath), repr(dirnames), repr(filenames)))
            #print(i)
            if dirpath is not self.imagePath:
                dirpath_components = dirpath.split("/")
                listImageTrain = []
                listLabelTrain = []
                listImageTest = []
                listLabelTest = []
                testFName = []
                trainFName = []
                semantic_label = dirpath_components[-1]
                
                _, label = os.path.split(semantic_label)

                #print("\nProcessing {}, {}".format(semantic_label,i))
                arr_Namelabel.append(label)
                self.count = 0
                train = 0
                test = 0

                for f in filenames:
                    #load images
                    file_path = os.path.join(dirpath, f)
                    #print(file_path)
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = rearrange(img, ' h w c ->  c h w ')

                    #arr_label.append(i-1)
                    # if mod append to test
                    if self.count % temp_mod == 0:
                        
                        listImageTest.append(img)
                        listLabelTest.append(i-1)
                        testFName.append(f)
                        test+= 1
                    # if not mod append to train
                    else:
                        listImageTrain.append(img)
                        listLabelTrain.append(i-1)
                        trainFName.append(f)
                        
                        train+= 1
                        
                    self.count+=1
        
                arrayImageTest = np.array(listImageTest, dtype='float64') / 255
                arrayImageTrain = np.array(listImageTrain, dtype='float64') / 255
                #print(np.array(arr_img).shape)
                arrayLabelTest = np.array(listLabelTest)
                arrayLabelTrain = np.array(listLabelTrain)
                
                arrayFNameTest = np.array(testFName)
                arrayFNameTrain = np.array(trainFName)
                
                self.labelName = np.array(arr_Namelabel)
                self.jum_kelas = len(self.labelName)

                if not hasattr(self, 'trainSet'):
                    self.trainSet = arrayImageTrain
                    self.trainLabel = arrayLabelTrain
                    self.testSet = arrayImageTest
                    self.testLabel = arrayLabelTest
                    
                    self.arrayFNameTrain = arrayFNameTrain
                    self.arrayFNameTest = arrayFNameTest
                else:
                    self.trainSet = np.concatenate((self.trainSet, arrayImageTrain), axis = 0)
                    self.trainLabel = np.concatenate((self.trainLabel, arrayLabelTrain), axis = 0)
                    self.testSet = np.concatenate((self.testSet, arrayImageTest), axis = 0)
                    self.testLabel = np.concatenate((self.testLabel, arrayLabelTest), axis = 0)
                    self.arrayFNameTest = np.concatenate((self.arrayFNameTest, arrayFNameTest), axis = 0)
                    self.arrayFNameTrain = np.concatenate((self.arrayFNameTrain, arrayFNameTrain), axis = 0)

        #print(self.arrayFNameTest)
        self.trainSet, self.trainLabel, self.arrayFNameTrain = self.unison_shuffled_copies_4(self.trainSet, self.trainLabel, self.arrayFNameTrain)
        #self.testSet, self.testLabel, self.arrayFNameTest  = self.unison_shuffled_copies_4(self.testSet, self.testLabel, self.arrayFNameTest)
        #print(self.arrayFNameTest)
        return self.trainSet, self.trainLabel, self.arrayFNameTrain ,self.testSet, self.testLabel, self.arrayFNameTest
    

    def loadtest(self,trainRatio=0.8,testRatio=0.2):
        temp_mod = math.ceil(trainRatio/testRatio)

        arr_Namelabel = []
        self.count = 0
        for i, (dirpath, dirnames, filenames) in tqdm(enumerate(os.walk(self.imagePath)), desc= "Loading Image Data"):
            #print('{} {} {}'.format(repr(dirpath), repr(dirnames), repr(filenames)))
            #print(i)
            if dirpath is not self.imagePath:
                dirpath_components = dirpath.split("/")

                listImageTest = []
                listLabelTest = []
                testFName = []
                semantic_label = dirpath_components[-1]
                
                _, label = os.path.split(semantic_label)

                #print("\nProcessing {}, {}".format(semantic_label,i))
                arr_Namelabel.append(label)
                self.count = 0
                train = 0
                test = 0

                for f in filenames:
                    #load images
                    file_path = os.path.join(dirpath, f)
                    #print(file_path)
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = rearrange(img, ' h w c ->  c h w ')

                    #arr_label.append(i-1)
                    # if mod append to test
                    if self.count % temp_mod == 0:
                        
                        listImageTest.append(img)
                        listLabelTest.append(i-1)
                        testFName.append(f)
                        test+= 1
                    # if not mod append to train
                    else:
                        pass
                        
                    self.count+=1
        
                arrayImageTest = np.array(listImageTest, dtype='float64') / 255

                #print(np.array(arr_img).shape)
                arrayLabelTest = np.array(listLabelTest)

                arrayFNameTest = np.array(testFName)

                
                self.labelName = np.array(arr_Namelabel)
                self.jum_kelas = len(self.labelName)

                if not hasattr(self, 'testSet'):

                    self.testSet = arrayImageTest
                    self.testLabel = arrayLabelTest
 
                    self.arrayFNameTest = arrayFNameTest
                else:

                    self.testSet = np.concatenate((self.testSet, arrayImageTest), axis = 0)
                    self.testLabel = np.concatenate((self.testLabel, arrayLabelTest), axis = 0)
                    self.arrayFNameTest = np.concatenate((self.arrayFNameTest, arrayFNameTest), axis = 0)

        #print(self.arrayFNameTest)
        #self.testSet, self.testLabel, self.arrayFNameTest  = self.unison_shuffled_copies_4(self.testSet, self.testLabel, self.arrayFNameTest)
        #print(self.arrayFNameTest)
        return self.testSet, self.testLabel, self.arrayFNameTest
                

    def loadGlcm(self,trainRatio=0.8,testRatio=0.2):
        
        """
        DOKUMENTASI 
        
        """
        
        temp_mod = math.ceil(trainRatio/testRatio)
        #arr_img = []
        #arr_label = []
        arr_Namelabel = []
        angless= [0., np.pi/4., np.pi/2., 3.*np.pi/4.]

        #angless= [0.]

        features = ['dissimilarity']

        #features = ['dissimilarity','energy','homogeneity','contrast']
        self.counter = 0
        for i, (dirpath, dirnames, filenames) in tqdm(enumerate(os.walk(self.imagePath)), desc= "Loading Image Data"):
            #print('{} {} {}'.format(repr(dirpath), repr(dirnames), repr(filenames)))
            #print(i)
            if dirpath is not self.imagePath:
                dirpath_components = dirpath.split("/")
                listGlcmTrain = []
                listLabelTrain = []
                listGlcmTest = []
                listLabelTest = []
                
                semantic_label = dirpath_components[-1]
                
                _, label = os.path.split(semantic_label)

                #print("\nProcessing {}, {}".format(semantic_label,i))
                arr_Namelabel.append(label)
                self.count = 0
                train = 0
                test = 0
                tempTrain = None
                tempTest = None

                for f in filenames:
                    #load images
                    file_path = os.path.join(dirpath, f)
                    #print(file_path)
                    gray = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                
                    glcm = greycomatrix(gray, distances=[5], angles=angless, levels=256, symmetric=True, normed=True)

                    temp_arr = None
                    
                    for feature in features:
                    
                        temp = greycoprops(glcm, feature)
                        if temp_arr is None:
                            temp_arr = temp
                        else:
                            temp_arr = np.concatenate((temp_arr, temp), axis=1)
                    
                    if self.count % temp_mod != 0:
                        
                        if tempTrain is None:
                            tempTrain = temp_arr
                            listLabelTrain.append(i-1)
                        else:
                            tempTrain = np.concatenate((tempTrain, temp_arr), axis=0)
                            listLabelTrain.append(i-1)
                        
                        
                        train+= 1
                        
                    # if not mod append to test
                    else:
                        if tempTest is None:
                            tempTest = temp_arr
                            listLabelTest.append(i-1)
                        else:
                            tempTest = np.concatenate((tempTest, temp_arr), axis=0)
                            listLabelTest.append(i-1)
                        test+= 1
                        
                        
                    self.count+=1
                    self.counter+=1
        
                
                
                self.labelName = np.array(arr_Namelabel)
                self.jum_kelas = len(self.labelName)

                if not hasattr(self, 'trainSet'):
                    self.trainSet = tempTrain
                    self.trainLabel = listLabelTrain
                    self.testSet = tempTest
                    self.testLabel = listLabelTest
                    
                    
                else:
                    self.trainSet = np.concatenate((self.trainSet, tempTrain), axis = 0)
                    self.trainLabel = np.concatenate((self.trainLabel, listLabelTrain), axis = 0)
                    
                    self.testSet = np.concatenate((self.testSet, tempTest), axis = 0)
                    self.testLabel = np.concatenate((self.testLabel, listLabelTest), axis = 0)
                    
                    
        print("counter", self.count, self.counter)
        self.trainSet, self.trainLabel = self.unison_shuffled_copies_2(self.trainSet, self.trainLabel)
        self.testSet, self.testLabel = self.unison_shuffled_copies_2(self.testSet, self.testLabel)
        return self.trainSet, self.trainLabel, self.testSet, self.testLabel

    def loadHistogram(self,trainRatio=0.8,testRatio=0.2):
        temp_mod = math.ceil(trainRatio/testRatio)

        arr_Namelabel = []
        

        self.counter = 0
        for i, (dirpath, dirnames, filenames) in tqdm(enumerate(os.walk(self.imagePath)), desc= "Loading Image Data"):
            #print('{} {} {}'.format(repr(dirpath), repr(dirnames), repr(filenames)))
            #print(i)
            if dirpath is not self.imagePath:
                dirpath_components = dirpath.split("/")
                listGlcmTrain = []
                listLabelTrain = []
                listGlcmTest = []
                listLabelTest = []
                
                semantic_label = dirpath_components[-1]
                
                _, label = os.path.split(semantic_label)

                #print("\nProcessing {}, {}".format(semantic_label,i))
                arr_Namelabel.append(label)
                self.count = 0
                train = 0
                test = 0
                tempTrain = None
                tempTest = None

                for f in filenames:
                    #load images
                    file_path = os.path.join(dirpath, f)

                    img = cv2.imread(file_path)
                    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

                    hist_h = cv2.calcHist( [hsv], [0], None, [180], [0, 180] )
                    hist_h = hist_h.T
                    
                    if self.count % temp_mod != 0:
                        
                        if tempTrain is None:
                            tempTrain = hist_h
                            listLabelTrain.append(i-1)
                        else:
                            tempTrain = np.concatenate((tempTrain, hist_h))
                            listLabelTrain.append(i-1)
 
                        train+= 1
                        
                    # if not mod append to test
                    else:
                        if tempTest is None:
                            tempTest = hist_h
                            listLabelTest.append(i-1)
                        else:
                            tempTest = np.concatenate((tempTest, hist_h))
                            listLabelTest.append(i-1)
                        test+= 1
                        
                        
                    self.count+=1
                    self.counter+=1
        
                
                
                self.labelName = np.array(arr_Namelabel)
                self.jum_kelas = len(self.labelName)

                if not hasattr(self, 'trainSet'):
                    self.trainSet = tempTrain
                    self.trainLabel = listLabelTrain
                    self.testSet = tempTest
                    self.testLabel = listLabelTest
                    
                    
                else:
                    self.trainSet = np.concatenate((self.trainSet, tempTrain), axis = 0)
                    self.trainLabel = np.concatenate((self.trainLabel, listLabelTrain), axis = 0)
                    
                    self.testSet = np.concatenate((self.testSet, tempTest), axis = 0)
                    self.testLabel = np.concatenate((self.testLabel, listLabelTest), axis = 0)
                    
                    
        print("counter", self.count, self.counter)
        self.trainSet, self.trainLabel = self.unison_shuffled_copies_2(self.trainSet, self.trainLabel)
        self.testSet, self.testLabel = self.unison_shuffled_copies_2(self.testSet, self.testLabel)
        return self.trainSet, self.trainLabel, self.testSet, self.testLabel






#mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
#workPath = os.path.split(mainPath) #path working folder (whole file project)
#imagePath = "data_jepun"
#data = Data(workPath,imagePath)
#trainSet, trainLabel, testSet, testLabel = data.load(trainRatio=0.8,testRatio=0.2)

#print("ts",trainSet.shape)
#print("tl",trainLabel.shape)
#print("tts",testSet.shape)
#print("ttl",testLabel.shape)


"""
"mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
workPath = os.path.split(mainPath) #path working folder (whole file project)
imagePath = "data_jepun"
data = Data(workPath,imagePath)
trainSet, trainLabelSet, testSet, testLabelSet = data.loadGlcm(trainRatio=0.8,testRatio=0.2)
print("ts",trainSet.shape)
print("tl",trainLabelSet.shape)
print("tts",testSet.shape)
print("ttl",testLabelSet.shape)
"""
