import torch 
import torch.nn as nn 
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split




class encode(nn.Module):
    def __init__(self,vocab_size,n_dim,n_heads,n_layer,d_ff):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,n_dim)
        self.pos_embedding=nn.Embedding(max_len,n_dim)
        encoder_layer=nn.TransformerEncoderLayer(d_model=n_dim,d_ff=d_ff,nhead=n_heads,batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=n_layer)
        self.norm=nn.LayerNorm(n_dim)
        self.fc=nn.Linear(d_model,vocab_size)
        self.norm=nn.LayerNorm(vocab_size)
    def forward(self,x):
        batch_size,seq_len=x.shape
        positions=torch.arrange(0,seq_len,device=x.device).unsqueeze(0)
        x=self.embedding(x)+self.pos_embedding(positions)
        x=self.encoder(x)
        x=self.norm(x)
        x=self.fc(x)
        logits=self.norm(x)
        return logits

#Creating the dataloader and dataset class 
class NewDataset(Dataset):
    def __init__(self,text_file,tokeniser):
        with open (text_file,'r') as file:
            self.text_file=file.read()
        self.tokeniser=tokeniser
        self.data=self.tokeniser(self.text_file)
    def __len__(self,data):
        return len(self.data)
        #Tokeniser returns a dict of  input_ids and attentiion mask
    def __getitem__(self,idx):
        return {"input_ids":self.data["input_ids"][idx], "attention_mask":self.data["attention_mask"][idx]}

dataset=NewDataset()
X_train,X_test,Y_train,Y_test=train_test_split(dataset,test_size=0.2)

train_dataloader= DataLoader(X_train,batch_size=32,shuffle=True)
test_dataloader= DataLoader(X_test,batch_size=32,shuffle=False)

epochs = 100 
optimiser = optim.Adam(encoder.parameters(),lr=learning_rate)
criterion=nnn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_features , batch_labels in train_dataloader :
        output=encoder(batch_features)
        loss=criterion(output,batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

#printing the loss function 
loss=nn.CrossEntropyLoss()
