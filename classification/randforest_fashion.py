import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from struct import unpack
from sklearn.ensemble import RandomForestClassifier

from time import time

#open ubyte files function
def openimage(path):
    f = open(path, 'rb')
    #4/4/4/4 : magic num, image numbers, row count, column count
    magic, num, rows, cols = unpack('>IIII', f.read(16))
    #uint8 = (unsigned)8bit
    image = np.fromfile(f, dtype=np.uint8).reshape(num, 28*28) #28*28 pixel
    f.close()
    return image

 
def openlabel(path):
    f = open(path, 'rb')
    #4/4 : magic num, label numbers
    magic, num = unpack('>II', f.read(8))
    label = np.fromfile(f, dtype=np.uint8)
    f.close()
    return label


img_train = openimage("data/train-images-idx3-ubyte") #training set images
label_train = openlabel("data/train-labels-idx1-ubyte") #training set labels
img_test = openimage("data/t10k-images-idx3-ubyte") #test set images
label_test = openlabel("data/t10k-labels-idx1-ubyte") #test set labels

print("Data Loading Complete")

#['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

classifier=RandomForestClassifier(n_estimators=100)

print("Train start")
start = time()
classifier.fit(img_train, label_train)
end = time()

time = end - start
pred = classifier.predict(img_test)

#Time and Accuracy Score
print('Train Time：%d분 %d초' % (time//60,time-60 * (time//60)))
print("Accuracy= ", accuracy_score(pred, label_test))
print(classification_report(label_test,pred))
