import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd      
import optuna



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



#creating the nn model 
class MLP(nn.Module):
    def __init__(self, num_features, hidden_layers, number_of_neurons, activations, dropout_rate):
        super().__init__()
        layers = []
        in_features = num_features
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_features, number_of_neurons))
            layers.append(nn.BatchNorm1d(number_of_neurons))
            
            # Select activation function
            if activations == "ReLU":
                layers.append(nn.ReLU())
            elif activations == "Tanh":
                layers.append(nn.Tanh())
            elif activations == "Softmax":
                layers.append(nn.Softmax(dim=1))
            elif activations == "Leaky ReLU":
                layers.append(nn.LeakyReLU())
                
            layers.append(nn.Dropout(dropout_rate))
            in_features = number_of_neurons

        layers.append(nn.Linear(number_of_neurons, 10))
        # Note: Softmax is removed because nn.CrossEntropyLoss expects raw logits.
        self.model = nn.Sequential(*layers)    

    def forward(self, X):
        return self.model(X)





# creating the hyperparameter traini object 

def objective(trial):

    hidden_layer=trial.suggest_int("hidden_layer",1,10)
    number_of_neurons=trial.suggest_int("number_of_neurons",100,500,step=100)
    activation_functions=trial.suggest_categorical("activations",["ReLU","Tanh","Softmax","Leaky ReLU"])
    optimiser_name=trial.suggest_categorical("optimiser",["RMSProp","Adam","SGD","Adagrad","AdamW"])
    learning_rate=trial.suggest_float("learning_rate",1e-5,1e-1,log=True)
    dropout_rate=trial.suggest_float("dropout_rate",0.1,0.5,step=0.1)
    batch_size=trial.suggest_categorical("batch_size",[16,32,64,128])
    epochs=trial.suggest_int("epochs",10,200,step=10)
    weight_decay=trial.suggest_float("weight_decay",1e-5,1e-1,log=True)    

    #creating dataloader
    TrainDataloader=DataLoader(TrainDataset,batch_size=batch_size,shuffle=True,pin_memory=True)
    TestDataloader=DataLoader(TestDataset,batch_size=batch_size,shuffle=False,pin_memory=True)
    
    #Model initialisation
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
    
    model.train()
    for epoch in range(epochs):
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
    return accuracy

study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler())
study.optimize(objective,n_trials=50)
print(f'Best Accuracy: {study.best_value}')
print(f'Best Parameters : {study.best_params}')


        











