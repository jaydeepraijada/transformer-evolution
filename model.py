import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #done in the paper, not sure why ("In the embedding layers, we multiply those weights by √dmodel.")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout =nn.Dropout(dropout)

        #create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # (seq_len, 1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2)
        #Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term) # (seq_len, d_model/2)
        #Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term) # (seq_len, d_model/2)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe) #register the positional encoding as a buffer, so it is not a parameter of the model

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False) #add the positional encoding to the input
        return self.dropout(x) #apply dropout to the input

class LayerNorm(nn.Module):
    def __init__(self,eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.aplha = nn.Parameter(torch.ones(1)) # multipled
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean)/ (std + self.eps) + self.bias

class FeedForwardBlock(nn.Moduel):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1 (cause bias by default is True)
        self.dropout = nn.Dropout(dropout) 
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2

    def forward(self, x):
        #(B, seq_len, d_model) --> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,  d_model:int, h: int, dropout: float) -> None: #h is the number of heads
        super().__init__()
        self.h = h
        self.d_model = d_model
        assert d_model % h == 0, "d_mdoel is not divisible by h"

        self.d_k = d_model//h

        self.w_q = nn.Linear(d_model, d_model) #W_query
        self.w_k = nn.Linear(d_model, d_model) #W_key 
        self.w_v = nn.Linear(d_model, d_model) #W_value

        self.w_o = nn.Linear(d_model, d_model) #W_output
        self.dropout = nn.Dropout(dropout)

    @staticmethod #you can call without instance of class
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # (Batch , h, seq_len, d_k) --> (Batch, h, Seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    



    def forward(self,q,k,v,mask=False):
        query = self.w_q(q) #(Batch, seq_len, d_model) @ (Batch, d_model,d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k) #(Batch, seq_len, d_model) @ (Batch, d_model,d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v) #(Batch, seq_len, d_model) @ (Batch, d_model,d_model) -> (Batch, seq_len, d_model)

        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, h, d_k) --> (Bathc, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.dk).transpose(1,2)  
        key = query.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], key.shape[1], self.h, self.d_k).tranpose(1,2)
 
        x, self.attention_scores  = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_len, d_k) --> (Batch, Seq_len, h, d_k) --> (Batch, Seq_len, d_model)
        x = x.transpose(1,2).contiguous.view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        return self.w_o(x)


    