import torch 
from torch import nn 
class Encoder(nn.Module):
    def __init__(self,d_model,nhead,dropout=0.1):
        super().__init__()