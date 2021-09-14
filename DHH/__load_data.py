import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

def load_data(arg):
    gen_temp = pd.read_csv('data\Genetic_alterations.csv',index_col=0)
    time_temp = pd.read_csv('data\Survival_time_event.csv',index_col=0)
    treat_temp = pd.read_csv('data\Treatment.csv',index_col=0)

    input_data=np.concatenate((gen_temp, time_temp), axis=1) #concatenated input data
    target_data=treat_temp.to_numpy()