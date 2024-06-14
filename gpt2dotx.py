from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # k, q, and v projections for all head *** in a batch ***
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # mask aka bias
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        # batch size, sequence length, and embeddings dimensionality (n_embed)
        B, T, C = x.size()

        # calculate q, k, and v values for all heads in batch and move forward head to be the batch
        # nh: number of head, hs: head size -> C = nh * hs (number of channels)
        # eg: in GPT-2, n_head = 12, hs = 64, so nh * hs = C = 768 (768 dimensions for embeddings) channels in transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        
        k = k.view((B, T, self.n_head, C // self.n_head)).transpose(1, 2) # (B, n_head, T, hs)
        q = k.view((B, T, self.n_head, C // self.n_head)).transpose(1, 2) # (B, n_head, T, hs)
        v = k.view((B, T, self.n_head, C // self.n_head)).transpose(1, 2) # (B, n_head, T, hs)
        
        # attention: materializes the large (T, T) matrix for all queries and keys
        att = (q @ k.transpose(-2, -1)) / (math.sqrt(k.size(-1)))


class MLP(nn.Module):  # aka FFN
    def __init__(self, config):

        # initialize the fully connected layer to 4 times the embedding size
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)

        # activation function for non-linear transformation
        self.gelu = nn.GELU()

        # convert the dimension of input back to the embedding size
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


# each block in transformer architecture
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):

        # Normalize before self-attention and add the result to the input
        x += self.attn(self.ln_1(x))
    
        # Normalize before MLP and add the result to the input
        x += self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    def __init__(self, block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embed=84):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer  # num transformer blocks
        self.n_head = n_head
        self.n_embed = n_embed


class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(
            {
                # Token embeddings
                "wte": nn.Embedding(config.vocab_size, config.n_embed),
                # Positional embeddings
                "wpe": nn.Embedding(config.block_size, config.n_embed),
                # Transformer blocks (hidden layers)
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # Final layer normalization
                "ln_f": nn.LayerNorm(config.n_embed),
            }
        )

        # projects the embedded vectors to size of vocabulary from tokenizer
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
