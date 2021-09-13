import numpy as np
import pandas as pd
#from sklearn.metrics import accuracy_score
from struct import unpack
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def pca_to_2d(train, test):
    print("before PCA shape = ", train.shape, test.shape)
    pca = PCA(n_components=2) # 784D -> 2D
    pca.fit(train)
    x_train = pca.transform(train)
    x_test = pca.transform(test)
    print("PCA output shape = ", x_train.shape, x_test.shape) #(60000, 2) (10000, 2)
    return x_train, x_test

img_train,img_test=pca_to_2d(img_train,img_test) 

#KMeans Model
kmeans = KMeans(n_clusters=10, n_init=100)
'''
x = img_train[:, 0]
y = img_train[:, 1]
plt.scatter(x, y, alpha=0.3)
plt.show()
'''
kmeans.fit(img_train)
label1= kmeans.predict(img_train)
label2 = kmeans.predict(img_test)

#Train Data _ Fitting Result
'''
x = img_train[:, 0]
y = img_train[:, 1]
plt.scatter(x, y, c=label1, alpha=0.3)
plt.show()
'''
#Test Data _ Fitting Result

x = img_test[:, 0]
y = img_test[:, 1]
plt.scatter(x, y, c=label2, alpha=0.3)
plt.show()

ct1=pd.crosstab(label1,label_train,colnames=['cloth '])
ct2=pd.crosstab(label2,label_test,colnames=['cloth '])
print('\n','====Train data====\n',ct1,'\n','====Test data====\n', ct2)
