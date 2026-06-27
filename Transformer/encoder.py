import torch 
import torch.nn as nn 




class encode(nn.Module):
    def __init__(self,vocab_size,n_dim,n_heads,n_layer,d_ff):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,n_dim)
        self.dropout=nn.Dropout(0.1)
        self.pos_embedding=nn.Embedding(max_len,n_dim)
        self.encoder_layer=nn.TransformerEncoderLayer(d_model=n_dim,d_ff=d_ff,nhead=n_heads,batch_first=True)
        
       
       
    def forward(self,x):
        return logits
        