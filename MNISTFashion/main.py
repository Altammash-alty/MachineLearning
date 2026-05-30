import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd      



torch.manual_seed(37)
device = torch.device('cuda' if  torch.cuda.is_available()else 'cpu')
print(f'Using Device : {device} ')

# ── Data Loading ──────────────────────────────────────────────────────────────
df = pd.read_csv("fashion-mnist_train.csv")

X = df.drop(columns=["label"]).values.astype(np.float32) / 255.0  # normalize to [0,1]
y = df["label"].values
#DataManipulation

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.16,random_state=37)

X_train=torch.tensor(X_train, dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.long)

#creating dataset 
class MyDataset(Dataset):
    def __init__(self,features,labels):
        self.features=features
        self.labels=labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self,idx):
        return self.features[idx] , self.labels[idx]

TrainDataset = MyDataset(X_train, y_train)
TestDataset = MyDataset(X_test,y_test)

#creating dataloader
TrainDataloader=DataLoader(TrainDataset,batch_size=128,shuffle=True,pin_memory=True)
TestDataloader=DataLoader(TestDataset,batch_size=128,shuffle=False,pin_memory=True)

    #creating the nn model 
    class MLP(nn.Module):
        def __init__(self,num_features):
            super().__init__()
            layers=[]
            for i in ranve
                      
                        )

        def forward(self,X):
        return self.model(X)





# creating the hyperparameter trainign object 

def objective(trial):

    hidden_layer=trial.suggest_int("hidden_layer",1,10)
    number_of_neurons=trial.suggest_int("number_of_neurons",100,500,step=100)
    activation_functions=trial.suggest_categorical("activations",["ReLU","Tanh","Softmax","Leaky ReLU"])
    optimiser=trial.suggest_categorical("optimiser",["RMSProp","Adam","SGD","Adagrad","AdamW"])
    learning_rate=trial.suggest_float("learning_rate",1e-5,1e-1,log=True)
    dropout_rate=trial.suggest_float("dropout_rate",0.0,0.5,step=0.1)
    batch_size=trial.suggest_categorical("batch_size",[16,32,64,128])
    epochs=trial.suggest_int("epochs",10,200,step=10)
    weight_decay=trial.suggest_float("weight_decay",1e-5,1e-1,log=True)    
    layers=[]

    #Learning Rate , Epochs
    lr=0.001
    epochs=100

    #Model initialisation
    model=MLP(X_train.shape[1])
    model.to(device)

    #Loss function 
    criterion=nn.CrossEntropyLoss()
    #Optimiser
    optimiser=optim.RMSprop(model.parameters(),lr=lr)

    print(model.parameters())


    #training loop
    for epoch in range(epochs):
        epoch_loss=0
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

            #upgrading the gradients
            optimiser.step()
            epoch_loss+=loss.item()
            avg_loss=epoch_loss/len(TrainDataloader)
        print("Epoch: ",epoch , "Loss: ", avg_loss)

    #Model Eval
    model.eval()

#Evaluation 
total=0
correct=0
with torch.no_grad():
    for batch_features,batch_labels in TestDataloader:
        batch_features=batch_features.to(device)
        batch_labels=batch_labels.to(device)        
        outputs   = model(batch_features)
        predicted = torch.argmax(outputs, dim=1)

        correct += torch.eq(predicted, batch_labels).sum().item()
        total   += batch_labels.size(0)
accuracy = correct/total

print(f'Accuracy obtained is : {accuracy}')




        











