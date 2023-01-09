import torch 
import math
from torch import nn
def get_attention(Q,K,V,dk,mask=None):
    scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(dk)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0,-1.e9)
    scores = nn.functional.softmax(scores,dim=-1)
    output = torch.matmul(scores,V)
    return output
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model//nhead
        self.h = nhead

        self.q_linear = nn.Linear(d_model,d_model)
        self.k_linear = nn.Linear(d_model,d_model)
        self.v_linear = nn.Linear(d_model,d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model,d_model)
    def forward(self,q,k,v, mask=None):
        # q: (bs,seq_len,d_model)
        bs = q.size(0)
        Q = self.q_linear(q).view(bs,-1,self.h,self.d_k) # bs,seq_len, h , d_k
        K = self.k_linear(k).view(bs,-1,self.h,self.d_k)
        V = self.v_linear(v).view(bs,-1,self.h,self.d_k)
        # transpose to get bs*h*seq_len*d_k
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        # get attention
        scores = get_attention(Q,K,V,self.d_k,mask=mask) # (bs,h,seq_len,d_k)
        concat = scores.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        output = self.out(concat) # (bs,seq_len,d_model)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model,d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)
    def forward(self,x):
        x = nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
class Norm(nn.Module):
    def __init__(self, d_model, eps =1.e-6):
        super().__init__()
        # create learnable parameters
        self.alpha = nn.parameter(torch.ones(d_model))
        self.bias = nn.parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self,x):
        norm = self.alpha*(x-x.mean(dim=-1,keepdims=True))
        norm = norm/(x.std(dim=-1,keepdims=True)+ self.eps)+ self.bias
        return norm

##===============================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model,nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attn = MultiHeadAttention(d_model,nhead,dropout)
        self.feedforward = FeedForward(d_model,d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self,x, mask, norm_first=True):
        if norm_first:
            # norm first
            xnorm = self.norm1(x)
            x  = x + self.dropout1(self.attn(xnorm,xnorm,xnorm,mask))
            xnorm = self.norm2(x)
            x  = x + self.dropout2(self.feedforward(xnorm))
        else:
            # norm after
            x  = x + self.dropout1(self.attn(x,x,x,mask))
            x = self.norm1(x)
            x  = x + self.dropout2(self.feedforward(x))
            x = self.norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model,nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)
        self.attn1 = MultiHeadAttention(d_model,nhead,dropout)
        self.attn2 = MultiHeadAttention(d_model,nhead,dropout)
        self.feedforward = FeedForward(d_model,d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self,x, e_outputs, src_mask, tgt_mask, norm_first=True):
        if norm_first:
            # norm first
            xnorm = self.norm1(x)
            x  = x + self.dropout1(self.attn1(xnorm,xnorm,xnorm,tgt_mask))
            xnorm = self.norm2(x)
            x  = x + self.dropout2(self.attn2(xnorm,e_outputs,e_outputs,src_mask))
            xnorm = self.norm3(x)
            x  = x + self.dropout3(self.feedforward(xnorm))
        else:
            # norm after
            x  = x + self.dropout1(self.attn1(x,x,x,tgt_mask))
            x = self.norm1(x)
            x  = x + self.dropout2(self.attn2(x,e_outputs,e_outputs,src_mask))
            x  = self.norm2(x)
            x  = x + self.dropout3(self.feedforward(x))
            x = self.norm3(x)
        return x