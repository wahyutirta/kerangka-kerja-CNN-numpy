from lenet5 import *
import numpy as np
import matplotlib.pyplot as plt

"""
file ini digunakan untuk menampilkan dan menyimpan feature map secara hard code
berikut susunan layer berdasarkan file lenet5
1 = feature map convo 1
2 = feature map max pool 1
3 = feature map convo 2
4 = feature map max pool 2
"""


def plotimage(imgs):
    # create figure
    fig = plt.figure(figsize=(4, 7))
    
    rows = 3
    columns = 2
    counter = 1
    
    
    for img in imgs:
        fig.add_subplot(rows, columns, counter)
        title = str("feature " + str(counter))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        counter += 1
        
    plt.legend()
    #plt.savefig('FMAP.png', dpi=300)

    plt.show()
        
        


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
Y_test = np.zeros((len_label, kelas))
Y_test[np.arange(len_label), testLabel[range(0, len_label)]] = 1
    
method = "adam"
epochs = 201
batch = 32
learningRate = 0.0001
    
mode = "test"
    
if mode == "train":
    mylenet = LENET5(X_train, Y_train, X_test, Y_test, method=method,epochs=epochs, batch=batch, learningRate=learningRate )
    
    
    layer_time = []
        
    start = timeit.default_timer()
    mylenet.lenet_train(method=method, epochs=epochs, batch=batch, learningRate=learningRate, zeta=0)
    stop = timeit.default_timer()
    print("Training time:", stop - start)
    print("Training ", end="")
    
    mylenet.save_parameters(mainPath)
    imgpath= "C:/Users/ASUS/Documents/py/cnn-numpy/data_jepun/bali/bali_(2).jpg"
    temp = os.path.split(imgpath)
    prob = mylenet.one_image(mylenet.layers, imgpath )
    print("\nFile Name ::", temp[1], " Tipe bunga ::", data.labelName[np.argmax(prob)], "||" ,
          "confidence ::", prob[0,np.argmax(prob)])
    acc, loss, time = mylenet.lenet_predictions(mylenet, mylenet.layers,X_test, Y_test, fNameTest, data.labelName)
    mylenet.printpred(acc, loss, time)

elif mode == "test":
    mylenet = LENET5([], [], [], [], method=method,epochs=epochs, batch=batch, learningRate=learningRate )
    imgpath= "C:/Users/ASUS/Documents/py/cnn-numpy/data_jepun/Plumeria_rubra_L_cendana/cendana_(1).jpg"
    temp = os.path.split(imgpath)
        
    """ load training history """
    mylenet.load_train_details(mainPath=mainPath,epochs=epochs,method=method, batch=batch, learningRate=learningRate )
    
    """ testing one image """
    print("Params: batch=", batch, " learning rate=", learningRate, "method=", method, "epochs=", epochs)
        
    mylenet.load_parameters(mainPath=mainPath,epochs=epochs,method=method, batch=batch, learningRate=learningRate)
    
        #acc, loss, time = mylenet.lenet_predictions(mylenet, mylenet.layers,X_test, Y_test,fNameTest, data.labelName)
        #mylenet.printpred(acc, loss, time)
    #prob = mylenet.one_image(mylenet.layers, imgpath )
    #print("\nFile Name ::", temp[1], " Tipe bunga ::", data.labelName[np.argmax(prob)], "||" ,
          #"confidence ::", prob[0,np.argmax(prob)])
        
    feature = mylenet.displayFeature(mylenet.layers, imgpath, 1)
    
    img = feature.astype(np.uint8)
    plotimage(img)
        

    
    