import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#npy file load
data = np.load('regression.npy')
data_t = np.load('regression_test.npy')

#define train data & test data
X_train=np.c_[data[:,0]]
y_train=np.c_[data[:,1]]
X_test=np.c_[data_t[:,0]]
y_test=np.c_[data_t[:,1]]

# Polynomial REGRESSION

#Create Model - quaternary function (PolynomialFeatures=4)
pipeline = make_pipeline(PolynomialFeatures(4), LinearRegression())
#Model Fitting
pipeline.fit(np.array(X_train), y_train)

#predicted data (for test data)
y_pred=pipeline.predict(X_test)

df = pd.DataFrame({'x': X_test[:,0], 'y': y_pred[:,0]})
df.sort_values(by='x',inplace = True)
points = pd.DataFrame(df).to_numpy()

#Model Evaluation - Accuracy
accuracy_score = pipeline.score(X_train,y_train)
print('Polynomial Model Accuracy: ', accuracy_score)

plt.scatter(X_test,y_test, color="black")
plt.plot(points[:, 0], points[:, 1],color="blue", linewidth=3)
plt.title("Polynomial Regression")

plt.show()
