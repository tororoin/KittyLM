import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MLP, Attention, LayerNorm

class KittyLMConfig:
    """
    Config according to the GPT-2 weights on huggingface.
    Using a vocab size that is a multiple of 64 to speed up the processing

    """
    block_size = 1024
    vocab_size = 50304 # 50257 in the original and hf implementation weights
    n_layer = 12
    n_heads = 12
    d_model = 768
    dropout = 0.0
    bias = True

class KittyLMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.preln = LayerNorm(config.d_model, bias = config.bias)
        self.attention = Attention(config)
        self.postln = LayerNorm(config.d_model, bias = config.bias)
        self.mlp = MLP(config)

    def forward(self, input):
        input = self.preln(input)
        input = self.attention(input)
        input = self.postln(input)
        output = self.mlp(input)
        return output
        # pass

class KittyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(num_embeddings = config.vocab_size, embedding_dim = config.d_model)
        self.position_embeddings = nn.Embedding(num_embeddings = config.block_size, embedding_dim = config.d_model)
        self.blocks = nn.ModuleList([KittyLMBlock(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.d_model, bias = config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias = False)

        # weight tying
        self.token_embeddings.weight = self.lm_head.weight

        #init weights
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith('projection.weight'):
                nn.init.normal_(parameter, mean = 0.0, std = 0.2 / math.sqrt(2 * config.n_layer))

        print(" parameter count : %.2fM" % (self._get_parameter_count(non_embedding = False) / 1e6))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.2)

    def _get_parameter_count(self, non_embedding = True):

        nparams = sum(param.numel() for param in self.parameters())
        if non_embedding:
            nparams -= self.position_embeddings.weight.numel()
        return nparams

    
    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.config.block_size, "Sequence length cannnot be greater than model capacity"

        token_embeddings = self.token_embeddings(input_ids)
        position_ids = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)

        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    
