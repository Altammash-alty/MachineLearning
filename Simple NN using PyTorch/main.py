import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split

torch.manual_seed(37)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device : {device}')                   


x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.2,random_state=37)

#Creating Dataset
class  myDataset(Dataset):
    def __init__(self,features,labels): 
        self.features=features
        self.labels=labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self,idx):
        
        return self.features[idx],self.labels[idx]

TrainingDataset = myDataset(x_train,y_train)
TestDataset = myDataset(x_test,y_test)

TrainingData = DataLoader(TrainingDataset,batch_size=32,shuffle=True )
TestData = DataLoader(TestDataset,batch_size=32,shuffle=False )


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


epochs =50
learning rate = 0.002

for epoch in range (epochs):
    for batch_features , batch_labels in TrainingData :
         

model=Model(features.shape[1])
#Model.forward(Features)
model(features)
print(Model.summary)
