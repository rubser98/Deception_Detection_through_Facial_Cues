import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#eredita classe Dataset

class DeceptionDataset(Dataset):
    def __init__(self,X,y,symmetry):
        self.X_data = torch.from_numpy(X)
        if not symmetry:
            self.X_data = self.X_data[:,:,:-2]
        print(self.X_data.shape)
        self.Y_data = torch.from_numpy(y)
        self.n_samples = self.X_data.shape[0]

    def __getitem__(self,index):
        return self.X_data[index],self.Y_data[index]
    
    def __len__(self):
        return self.n_samples
       
                
