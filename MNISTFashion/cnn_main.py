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


device = ("cuda" if torch.cuda.is_available() else "cpu")

X=X.reshape(-1,1,28,28)
X=X.expand(-1,3,28,28)
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

TrainingDataLoader = DataLoader(TrainingDataset,batch_size=32,shuffle=True)
TestDataLoader = DataLoader(TestDataset,batch_size=32,shuffle=False)

class CNN(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,16,3,1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10),
        )
    def forward(self,X):
        image_features=self.conv(X)
        final_output=self.classifier(image_features)
        return final_output



model=CNN(num_features=X_train.shape[1])
model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0037)
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
            
        #upgrading the gradients
        optimiser.step()
        epoch_loss += loss.item()
            
        # Print progress every 5 epochs, or first and last epoch
        if epoch == 0 or (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            print(f"  Trial {trial.number} | Epoch {epoch+1:02d}/{epochs:02d} | Avg Loss: {epoch_loss/len(TrainDataloader):.4f}")

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
                return accuracy
print(f'Accuracy obtained is : {accuracy}')

