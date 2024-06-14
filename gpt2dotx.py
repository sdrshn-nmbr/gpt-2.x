from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CasualSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # k, q, and v projections for all heads *** in a batch ***
        self.c_attn: nn.Linear = nn.Linear(config.n_embed, 3 * config.n_embed)

        # output projection
        self.c_proj: nn.Linear = nn.Linear(config.n_embed, config.n_embed)

        # regularization
        self.n_head: int = config.n_head
        self.n_embed: int = config.n_embed

        # mask aka bias
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, and embeddings dimensionality (n_embed)
        B: int
        T: int
        C: int
        B, T, C = x.size()

        # calculate q, k, and v values for all heads in batch and move head forward to be the batch
        # nh: number of heads, hs: head size -> C = nh * hs (number of channels)
        # eg: in GPT-2, n_head = 12, hs = 64, so nh * hs = C = 768 (768 dimensions for embeddings) channels in transformer
        qkv: torch.Tensor = self.c_attn(x)
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q, k, v = qkv.split(self.n_embed, dim=2)

        k = k.view((B, T, self.n_head, C // self.n_head)).transpose(
            1, 2
        )  # (B, n_head, T, hs)
        q = q.view((B, T, self.n_head, C // self.n_head)).transpose(
            1, 2
        )  # (B, n_head, T, hs)
        v = v.view((B, T, self.n_head, C // self.n_head)).transpose(
            1, 2
        )  # (B, n_head, T, hs)

        # attention: materializes the large (T, T) matrix for all queries and keys
        att: torch.Tensor = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=2)

        # (B, nh, T, T) x (B, n_head, T, hs) -> (B, n_head, T, hs)
        y: torch.Tensor = att @ v

        # output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):  # aka FFN
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()

        # initialize the fully connected layer to 4 times the embedding size
        self.c_fc: nn.Linear = nn.Linear(config.n_embed, 4 * config.n_embed)

        # activation function for non-linear transformation
        self.gelu: nn.GELU = nn.GELU()

        # convert the dimension of input back to the embedding size
        self.c_proj: nn.Linear = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# each block in transformer architecture
class Block(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.ln_1: nn.LayerNorm = nn.LayerNorm(config.n_embed)
        self.attn: CasualSelfAttention = CasualSelfAttention(config)
        self.ln_2: nn.LayerNorm = nn.LayerNorm(config.n_embed)
        self.mlp: MLP = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize before self-attention and add the result to the input
        x = x + self.attn(self.ln_1(x))

        # Normalize before MLP and add the result to the input
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence/conext length

    # num tokens possible: 50_000 BPE merges + 256 byte tokens + 1 <|endoftext|>
    vocab_size: int = 50257

    n_layer: int = 12  # number of layers

    # number of heads (individual attention mechanisms within a multi-head attention module)
    # each head independently performs self-attention
    n_head: int = 12

    n_embed: int = 768  # embedding dimensions


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config: GPTConfig = config
        self.transformer: nn.ModuleDict = nn.ModuleDict(
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

        # projects the embedded vectors to the size of vocabulary from tokenizer
        self.lm_head: nn.Linear = nn.Linear(
            config.n_embed, config.vocab_size, bias=False
        )
