import torch 
import torch.nn as nn 
import torch.optim as optim
from sklearn.preprocesssing import train_test_split
import numpy as np 



#Data Loading 


#DataManipulation

X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.16,random_state=37)

X_train=torch.tensor(X.train)
y_train=torch.tensor(y.train)
X_test=torch.tensor(X.test)
y_test=torch.tensor(y.test)

#creating dataset 
class MyDataset(Dataset):
    def __init__(self,features,labels):
        self.features=features
        self.labels=labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self,idx):
        return self.features[idx] , self.labels[idx]

TrainDataset = MyDataset(X-train, y_train)
TestDataset = MyDataset(X_test,y_test)

#creating 









