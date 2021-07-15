import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from struct import unpack
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import argparse
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--randf', type=str )
args = parser.parse_args()

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


img_train = openimage("C:\/Users/suew1/Documents/git/jh-git/data/train-images-idx3-ubyte") #training set images
label_train = openlabel("C:\/Users/suew1/Documents/git/jh-git/data/train-labels-idx1-ubyte") #training set labels
img_test = openimage("C:\/Users/suew1/Documents/git/jh-git/data/t10k-images-idx3-ubyte") #test set images
label_test = openlabel("C:\/Users/suew1/Documents/git/jh-git/data/t10k-labels-idx1-ubyte") #test set labels

print("Data Loading Complete")

#['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

classifiers = {
    'RF' : RandomForestClassifier(n_estimators=50),
    'RF1' : RandomForestClassifier(n_estimators=100)
}

print("Train start")
classifiers[args.randf].fit(img_train, label_train)


pred = classifiers.predict(img_test)
print(args.randf)

# Accuracy Score
print("Accuracy= ", accuracy_score(pred, label_test))
print(classification_report(label_test,pred))
