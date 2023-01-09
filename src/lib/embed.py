import torch
from torch import nn, Tensor
import math

class Embedder(nn.Module):
    def __init__(self,vocab_size:int, d_model:int):
        super().__init__()
        self.d_model = d_model
        self. embed = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        x = self.embed(x)*math.sqrt(self.d_model)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int,max_len: int =2000, dropout: float =0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000)/d_model))
        # create pe matrix
        pe = torch.zeros(max_len,d_model)
        pe[:,0::2]= torch.sin(position*div_term)
        pe[:,1::2]= torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x:Tensor)-> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size()[-2]
        pe = torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)

