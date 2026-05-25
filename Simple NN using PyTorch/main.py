import torch 
import torch.nn as nn
import numpy as np

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
