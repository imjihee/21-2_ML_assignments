"""
def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

struct.unpack(format, buffer)
포맷 문자열 format에 따라 버퍼 buffer(아마도 pack(format, ...)으로 패킹 된)에서 언 패킹 합니다.
정확히 하나의 항목을 포함하더라도 결과는 튜플입니다.
바이트 단위의 버퍼 크기는 (calcsize()에 의해 반영되는) 포맷이 요구하는 크기와 일치해야 합니다.

"""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from struct import unpack

#open ubyte files function
def openimage(path):
    f = open(path, 'rb')
    #4/4/4/4 : magic num, image numbers, row count, column count
    magic, num, rows, cols = unpack('>IIII', f.read(16))
    #uint8 = (unsigned)8bit
    image = np.fromfile(f, dtype=np.uint8).reshape(num, 28*28) #28*28 pixel
    f.close()
    print(image)
    
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

svm=SVC()
#dictionary, Gaussian Kernel
params = {'kernel':['rbf'], 'C':[1]}
print("Train start")

#그리드를 사용한 복수 하이퍼 파라미터 최적화
#estimator=svm, param_grid=parameters
classifier=GridSearchCV(svm,params,n_jobs=2)

# img_train : X, label_train : y
classifier.fit(img_train, label_train)

pred = classifier.predict(img_test)

#Accuracy Score
print("accuracy= ", accuracy_score(pred, label_test))
