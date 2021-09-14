import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold

import pdb

def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

class CustomDataset(Dataset):
    def __init__(self,input_array,target_array,transform=None):
        self.data=[]
        
        for idx in range(1000):
            temp_target=target_array[idx]
            temp_input=input_array[idx][0:] ## erase outer []
            self.data.append([temp_input,temp_target])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        x,y=self.data[idx]
        sample={"x":x, "y":int(y)}
        return sample
    
    
class SimpleConvNet(nn.Module):
    '''
    Simple Convolutional Neural Network
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(302, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10,2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
  
if __name__ == '__main__':
  
  # Configuration options
    k_folds = 10
    num_epochs = 1 #need to be modified
    loss_function = nn.CrossEntropyLoss()
    
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)

    gen_temp = pd.read_csv('data\Genetic_alterations.csv',index_col=0)
    time_temp = pd.read_csv('data\Survival_time_event.csv',index_col=0)
    treat_temp = pd.read_csv('data\Treatment.csv',index_col=0)
    
    input_data=np.concatenate((gen_temp, time_temp), axis=1) #concatenated input data
    target_data=treat_temp.to_numpy()
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True) #k_folds = 10, make Kfold class
    
    # Start print
    print('--------------------------------')
    dataset=CustomDataset(input_data,target_data)
    
    # K-fold Cross Validation model evaluation // train_ids, test_ids : index of each data
    #enumerate 결과로 인해서, kfold.split function에 의해 인덱스가 출력되고 
    #해당 인덱스들에 fold라는 번호가 매겨진다.
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # indicis 출력받아서 for문으로 돌린다.
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        #train_ids: train index 900
        #print("test_ids in each fold: ", np.shape(test_ids)) #100개
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        #type : <class 'torch.utils.data.sampler.SubsetRandomSampler'>
        #Dataset 은 샘플과 정답(label)을 저장하고,
        #DataLoader 는 Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌉니다.
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=10, sampler=train_subsampler) #mini batch 900
        # _subsampler: contains index
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10, sampler=test_subsampler) #mini batch 100
        # Init the neural network
        network = SimpleConvNet()
        network.apply(reset_weights)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        
        #▶TRAINING◀
        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):
            
            # Print epoch
            print(f'Starting epoch {epoch+1}')
            
            # Set current loss value
            current_loss = 0.0
            
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                # Get inputs adn targets
                inputs = data['x'].float()
                targets = data['y']                
                # data : dictionary. x:input tensor, y:target tensor

                # Zero the gradients
                optimizer.zero_grad()
        
                # Perform forward pass
                #pdb.set_trace()
                outputs = network(inputs)
        
                # Compute loss
                loss = loss_function(outputs, targets)
        
                # Perform backward pass
                loss.backward()
        
                # Perform optimization
                optimizer.step()
        
                # Print statistics
                current_loss += loss.item()
                if i % 10 == 9:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 500))
                    current_loss = 0.0
            
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')
    
        # Saving the model
        save_path = f'model/fc/model-fold-{fold}.pth'
        torch.save(network.state_dict(), save_path)
        
        #▶TEST (EVALUATION)◀
        # Evaluationfor this fold USING TEST FOLD!!!!!
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                # inputs, targets = data
                inputs = data['x'].float()
                targets = data['y']

                # Generate outputs
                outputs = network(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
    
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
        print(f'Average: {sum/len(results.items())} %')