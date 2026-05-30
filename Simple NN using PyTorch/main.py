import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split

torch.manual_seed(37)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device : {device}')                   
    

#Crreating Dataset
class  TrainingDataset(Dataset):
    def __init__(self,features,labels): 
        self.features=features
        self.labels=labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx]




class Model(nn.Module):
    def __init__(self,num_features):
        super().__init__()
       # self.weights = torch.rand(num_features,requires_grad=True)
       # self.bias=torch.rand(1)
        self.network =nn.Sequential(
            nn.Linear(num_features,3),
            nn.ReLU(),
            nn.Linear(3,1),
            nn.Sigmoid()
        )
        

    def forward(self,features):
        out=self.network(features) 
        return out


features=torch.rand(10,5)

model=Model(features.shape[1])
#Model.forward(Features)
model(features)
print(Model.summary)
