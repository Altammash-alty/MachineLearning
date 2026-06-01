import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd      
import optuna

torch.manual_seed(37)
#Data Loading and Preprocessing
df = pd.read_csv("fashion-mnist_train.csv")
X=df.drop("label",axis=1).values.astype(np.float32)
y=df["label"].values

X=X.reshape(-1,1,28,28)
X=X/255.0

X_train,X_test,y_train,y_test=test_train_split(X,y,test_size=0.16,random_state=37)

class MyDataset(Dataset):
    def __init__(self,features,labels):
        self.features=features
        self.labels=labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self,index):
        return self.features[index],self.labels[index]  

TrainingDataset = MyDataset(X_train,y_train)
TestDataset=MyDataset(X_test,y_test)

TrainingDataLoader = DataLoader(TrainingDataset,batch_size=,shuffle=True)
TestDataLoader = DataLoader(TestDataset,batch_size=1,shuffle=False)

model=MLP(
        num_features=X_train.shape[1],
        hidden_layers=hidden_layer,
        number_of_neurons=number_of_neurons,
        activations=activation_functions,
        dropout_rate=dropout_rate
    )
    model.to(device)

    criterion=nn.CrossEntropyLoss()

    if optimiser_name=="RMSProp":
        optimiser=optim.RMSprop(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    elif optimiser_name=="Adam":
        optimiser=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    elif optimiser_name=="SGD":
        optimiser=optim.SGD(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    elif optimiser_name=="Adagrad":
        optimiser=optim.Adagrad(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    else :
        optimiser=optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    
    print(f"\n[Trial {trial.number}] Starting training with epochs={epochs}, batch_size={batch_size}...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_features , batch_labels in TrainDataloader:
            batch_features=batch_features.to(device)
            batch_labels=batch_labels.to(device)

            #Forward Pass 
            outputs=model(batch_features)

            #Calculate loss
            #zero the gradients
            optimiser.zero_grad()
            loss=criterion(outputs,batch_labels)

            #BackPass
            loss.backward()