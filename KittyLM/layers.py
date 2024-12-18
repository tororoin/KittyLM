import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    #pass 
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4*config.d_model, bias = config.bias)
        self.c_proj = nn.Linear(4*config.d_model, config.d_model, bias = config.bias)
        self.activation = nn.GELU() # avoid sudden zeroout of gradients and have a smoother actovation 
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        return self.dropout(self.c_proj(self.activation(self.c_fc(input))))
        

class Attention(nn.Module):
    #pass
    def __init__(self, config):
        super(Attention, self).__init__()
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias = config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias = config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias = config.bias)
        
        # final projection after attention
        self.projection = nn.Linear(config.d_model, config.d_model, bias = config.bias)

        # these are self-explanatory
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.head_size = self.d_model // self.n_heads

        self.register_buffer(
            'causal_mask', 
            torch.tril(torch.ones(config.block_size, config.block_size)) # create a block_size * block_size mask
            .view(1, 1, config.block_size, config.block_size) # add singletons so that shape is B * nh * block_size * block_size
        )

    def forward(self, input):
        B, T, D = input.size() # batch, length, dimension

        # reshape q,k,v to (B, nh, T, hs) from (B, T, D) -> (B, T, nh, hs) -> (B, nh, T, hs)
        # view shouldnt be used to transpose / permute as it messes up the data. chain a 
        # seperate transpose operation to transpose the the sequence length and head dimensions 
        # q, k, v  = self.c_attn(input).split(self.d_model, dim=2)
        q = self.q_proj(input).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(input).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(input).view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        # lets manually compute the attention score without einsum
        e = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        e = e.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))  # masking only the actual inportant information across sequencelength and head dimension
        alpha = F.softmax(e, dim = -1)
        alpha = self.attention_dropout(alpha)
        attention = alpha @ v
        attention = attention.transpose(1, 2).contiguous().view(B, T, D) # hstack all heads
        attention = self.projection(attention)
        attention = self.residual_dropout(attention)

        return attention


class LayerNorm(nn.Module):
    #pass
    def __init__(self, d_model, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        
        self.bias = nn.Parameter(torch.ones(d_model)) if bias else None

    def forward(self, input):
        ln = F.layer_norm(
            input = input,
            normalized_shape = self.weight.shape,
            weight = self.weight,
            bias = self.bias
        )
        return ln


