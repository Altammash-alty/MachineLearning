import torch 
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
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

#creating dataloader
TrainDataloader=DataLoader(TrainDataset,batch_size=32,shuffle=True)
TestDataloader=DataLoader(TestDataset,batch_sizesize=32,shuffle=False)

#creating the nn model 
class MLP(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(num_features,400),
            nn.ReLU(),
            nn.Linear(400,250),
            nn.Tanh(),
            nn.Linear(250,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.Softmax()
                    )

    def forward(self,X):
       return self.model(X)


#Learning Rate , Epochs
lr=0.001
epochs=200

#Model initialisation
model=MLP(X_train.shape[1])

#Loss function 
loss=nn.CrossEntropyLoss()
#Optimiser
optimiser=optim.RMSprop(model.parameters(),lr=lr)

print(model.parameters())


#training loop
for epochs in range(epochs):
    epoch_loss=0
    for batch_features , batch_labels in TrainDataloader:

        #Forward Pass 
        outputs=model(batch_features)

        #Calculate loss
        #zero the gradients
        optimiser.zero_grad()
        loss=loss(outputs,batch_labels)

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

for batch_features,batch_labels in TestDataloader:
    outputs=model(batch_features)
    predicted=torch.max(output,1)
    if (predicted==batch_labels):
        correct+=1
    total=total+batch_labels.shape[0]
accuracy = correct/total

print(f'Accuracy obtained is : {accuracy}')


        











