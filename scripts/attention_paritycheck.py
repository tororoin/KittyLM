import torch
from torch.nn import MultiheadAttention
from torch import nn
from layers import Attention
from model import KittyLMConfig

def parity_check_attn(config, input_B, input_T):

    # create random input tensor
    B, T, dim, n_heads = input_B, input_T, config.d_model, config.n_heads
    input_tensor = torch.randn(B, T, dim)

    # Calculate attention on input tensor using custom implemented attention class 
    attention_layer = Attention(config)
    custom_output = attention_layer(input_tensor)

    # Calculate attention using torch.nn.MultiheadAttention
    # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=config.dropout, bias=config.bias, batch_first=True )
    query = input_tensor.view(B, T, dim)
    key = query.clone()
    value = query.clone()

    attn_output, attn_output_weights = multihead_attn(query, key, value)

    assert attn_output.size() == custom_output.size(), f"custom attn output and pytorch attn output not same size: {custom_output.size()} vs. {attn_output.size()}"
    
    diff = torch.mean(torch.abs(custom_output -  attn_output))

    return "diff btwn custom implemented and pytorch multihead attn", diff.item()
    
print(parity_check_attn(KittyLMConfig, 1, 10))

