import torch 
import torch.nn as nn 




class encode(nn.Module):
    def __init__(self,vocab_size,n_dim,n_heads,n_layer,d_ff):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,n_dim)
        self.dropout=nn.Dropout(0.1)
       
    def forward(self,x):
        return logits
        