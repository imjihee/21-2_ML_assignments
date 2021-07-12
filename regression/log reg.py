import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#npy file load
data = np.load('regression.npy')
data_t = np.load('regression_test.npy')

#define train data & test data
X_train=np.c_[data[:,0]]
y_train=np.c_[data[:,1]]
X_test=np.c_[data_t[:,0]]
y_test=np.c_[data_t[:,1]]

# LOGISTIC REGRESSION
