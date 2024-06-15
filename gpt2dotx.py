from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

device = "cuda"

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embed)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):  # aka FFN
    def __init__(self, config: "GPTConfig"):
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
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.ln_1: nn.LayerNorm = nn.LayerNorm(config.n_embed)
        self.attn: CausalSelfAttention = CausalSelfAttention(config)
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
    def __init__(self, config: GPTConfig):
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

    def forward(self, idx, targets=None):
        #! idx is of shape (B, T)
        B, T = idx.size()

        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and posiition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embed)

        # part where you add the (learnable) token embeddings with the (fixed) position embeddings
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        # loss = None
        # if targets is not None:
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # return logits, loss
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained model weights from HF"""

        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embed are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M params
        }[model_type]

        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


model = GPT.from_pretrained("gpt2-xl")
# model = GPT(GPTConfig)
print("didnt crash haha")
model.eval()
model.to(device)

# prefix tokens
import tiktoken

num_return_sequences = 30
max_length = 20

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I am a language model, ")
print(len(tokens))
tokens = torch.tensor(tokens, dtype=torch.long)
print(tokens.size())
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

with open('fineweb.txt', 'r') as f:
    text = f.read()
    
