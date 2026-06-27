import torch 
import torch.nn as nn 




class encode(nn.Module):
    def __init__(self,vocab_size,n_dim,n_heads,n_layer,d_ff):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,n_dim)
        self.pos_embedding=nn.Embedding(max_len,n_dim)
        encoder_layer=nn.TransformerEncoderLayer(d_model=n_dim,d_ff=d_ff,nhead=n_heads,batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=n_layer)
        self.fc=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        x=self.embedding(x)
        x=self.pos_embedding(x)
        x=self.encoder(x)
        logits=self.fc(x)
        return logits
        