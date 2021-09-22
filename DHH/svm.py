import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import pdb
  
if __name__ == '__main__':
  
  # Configuration options
    k_folds = 10
    loss_function = nn.CrossEntropyLoss()
    
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)

    gen_temp = pd.read_csv('data\Genetic_alterations.csv',index_col=0)
    time_temp = pd.read_csv('data\Survival_time_event.csv',index_col=0)
    clinic_temp= pd.read_csv('data\Clinical_Variables.csv',index_col=0)
    treat_temp = pd.read_csv('data\Treatment.csv',index_col=0)
    
    input_data=np.concatenate((gen_temp, time_temp, clinic_temp), axis=1) #concatenated input data
    target_data=treat_temp.to_numpy()
    #pdb.set_trace()
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True) #k_folds = 10, make Kfold class

    acc=[]
    for fold, (train_ids, test_ids) in enumerate(kfold.split(input_data)):
        # indicis 출력받아서 for문으로 돌린다.
        # Print
        print('--------------------------------')
        print(f'FOLD {fold}')
        print('--------------------------------')
        X_train, X_test=input_data[train_ids], input_data[test_ids]
        Y_train, Y_test=target_data[train_ids], target_data[test_ids]
        Y_train = np.ravel(Y_train, order='C')
        Y_test = np.ravel(Y_test, order='C')
        
        #▶TRAINING◀
        svm=SVC()
        params = {'kernel':['rbf'], 'C':[10]} #poly - degree 2 or 3 // rbf - gamma 0.1 or 0.2
        print("Train start")
        classifier=GridSearchCV(svm,params,n_jobs=2)
        classifier.fit(X_train, Y_train)

        pred = classifier.predict(X_test)
        # Process is complete.
        print(f'K-FOLD RESULTS FOR {fold} FOLDS')
        acc_t=accuracy_score(pred, Y_test)
        print("accuracy= ", acc_t)
        acc.append(acc_t)
    
    avg_acc=sum(acc)/k_folds
    print("Average Accuracy is : %0.3f" %avg_acc )