import numpy as np
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#npy file load
data = np.load('regression.npy')
data_t = np.load('regression_test.npy')

#define train data & test data
X_train=np.c_[data[:,0]]
y_train=np.c_[data[:,1]]
X_test=np.c_[data_t[:,0]]
y_test=np.c_[data_t[:,1]]

#Create Model
lin_reg = LinearRegression()
#Model fitting
lin_reg.fit(X_train,y_train)

#predicted data (for test data)
pred = lin_reg.predict(X_test)

#model evaluation
accuracy_score = lin_reg.score(X_train,y_train)

#Print Model Accuracy
print('Model Accuracy: ', accuracy_score)


plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, pred, color='blue', linewidth=3)
plt.title("Linear Regression")

plt.show()
