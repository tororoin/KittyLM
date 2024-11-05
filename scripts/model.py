import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MLP, Attention, LayerNorm

class KittyLMBlock(nn.Module):
    pass

class KittyLMConfig:
    """
    Config according to the GPT-2 weights on huggingface.
    Using a vocab size that is a multiple of 64 to speed up the processing

    """
    block_size = 1024 # maximum length of input sequence (i.e. 1024 tokens)
    vocab_size = 50304 # 50257 in the original and hf implementation weights
    n_layer = 12
    n_heads = 12 # attn heads
    d_model = 768
    dropout = 0.0
    bias = True