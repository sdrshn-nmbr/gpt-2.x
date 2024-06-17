from dataclasses import dataclass
import math, time, inspect, os

import tiktoken

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed import init_process_group, destroy_process_group

# ! setup DDP (Distributed Data Parallel)
# * torchrun command sets env vars RANK, LOCAL_RANK, and WORLD_SIZE

ddp = int(os.environ.get("RANK"), -1) != -1
if ddp:
    # ! DDP reqiuires cuda as of right now
    assert torch.cuda.is_available(), "need cuda for DDP"

    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing, etc.

else:  # vanilla run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cuda"


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embed % config.n_head == 0
        # * key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # * output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # * regularization
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
        B, T, C = (
            x.size()
        )  # * batch size, sequence length, embedding dimensionality (n_embed)
        # * calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # * nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # * e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # * (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # * (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # * (B, nh, T, hs)

        # ____________________________________________________________________________ Will give OOM if B=16
        # attention: materializes the large (T, T) matrix for all queries and keys
        # att: torch.Tensor = (q @ k.transpose(-2, -1)) / (1.0 * math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # # (B, nh, T, T) x (B, n_head, T, hs) -> (B, n_head, T, hs)
        # y: torch.Tensor = att @ v
        # ____________________________________________________________________________

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # * flash attention

        # ____________________________________________________________________________

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # * re-assemble all head outputs side by side
        # * output projection

        y = self.c_proj(y)
        return y


class MLP(nn.Module):  # * aka FFN
    def __init__(self, config):
        super(MLP, self).__init__()

        # * initialize the fully connected layer to 4 times the embedding size
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)

        # * activation function for non-linear transformation
        self.gelu = nn.GELU()

        # * convert the dimension of input back to the embedding size
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# * each block in transformer architecture
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        # * Normalize before self-attention and add the result to the input
        x += self.attn(self.ln_1(x))

        # * Normalize before MLP and add the result to the input
        x += self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # * max sequence/conext length

    # * num tokens possible: 50_000 BPE merges + 256 byte tokens + 1
    vocab_size: int = 50257

    n_layer: int = 12  # * number of layers

    # * number of heads (individual attention mechanisms within a multi-head attention module)
    # * each head independently performs self-attention
    n_head: int = 12

    n_embed: int = 768  # * embedding dimensions


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()

        self.config = config
        self.transformer = nn.ModuleDict(
            {
                # * Token embeddings
                "wte": nn.Embedding(config.vocab_size, config.n_embed),
                # * Positional embeddings
                "wpe": nn.Embedding(config.block_size, config.n_embed),
                # * Transformer blocks (hidden layers)
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # * Final layer normalization
                "ln_f": nn.LayerNorm(config.n_embed),
            }
        )

        # * projects the embedded vectors to the size of vocabulary from tokenizer
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # * weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):

                # * 2 times num layers for 1 attn and 2 mlp/ffn -> done to maintain std of 1 for weights/tensors | otherwise grows inside residual stream
                std *= (2 * self.config.n_layer) ** -0.5

            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # * * idx is of shape (B, T)
        B, T = idx.size()

        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # * forward the token and posiition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # * shape (T)
        pos_emb = self.transformer.wpe(
            pos
        )  # * position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(
            idx
        )  # * token embeddings of shape (B, T, n_embed)

        # * part where you add the (learnable) token embeddings with the (fixed) position embeddings
        x = tok_emb + pos_emb

        # * forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # * forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # * (B, T, vocab_size)
        loss = None
        if targets is not None:
            # * flattening out (B, T, vocab_size) into 2 dims -> (inferred, vocab size) = (B * T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # * start with all the candidate params that require decay
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # * create optim groups. any param with dim >= 2 will be weight decayed - else no
        # * i.e: all weight tensors in matmuls + embeddings decay, all biases and layernorms dont

        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(
            f"num decayed param tensors: {len(decay_params)} with {num_decay_params} params"
        )
        print(
            f"num non-decayed param tensors: {len(nodecay_params)} with {num_nodecay_params} params"
        )

        # ! AdamW optimizer BUT with fuse if available (cuda only)
        fused_avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
        )

        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained model weights from HF"""

        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # * n_layer, n_head and n_embed are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),  # * 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # * 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),  # * 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),  # * 1558M params
        }[model_type]

        config_args["vocab_size"] = 50257  # * always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # * always 1024 for GPT model checkpoints

        # * create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # * discard this mask / buffer, not a param

        # * init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # * copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # * ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # * same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # * basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # * this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # * special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # * vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class DataloaderLite:
    def __init__(self, B, T):  # * * again, B is num batches and T is sequence length
        self.B, self.T = B, T

        with open("fineweb.txt", "r", encoding="utf-8") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)

        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # * state
        self.curr_pos = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.curr_pos : self.curr_pos + B * T + 1]
        x = buf[:-1].view(B, T)  # * inputs
        y = buf[1:].view(B, T)  # * targets

        # * advance position in the tensor
        self.curr_pos += B * T

        # * if next batch causes out-of-bound reset to beginning
        if self.curr_pos + (B * T + 1) > len(self.tokens):
            self.curr_pos = 0

        return x, y


# ! ____________________________________________________________________________________________________


torch.cuda.manual_seed(1337)

total_B = 2**19  # approx 0.5 million (as used in GPT-3 paper)
B = 8  # micro batch size
T = 1024
assert (
    total_B % (B * T * ddp_world_size) == 0
), "make sure total battch size is divisble by batch * seq lens * num of gpus"

grad_accum_steps = total_B // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_B}")
    print(f"calculated gradient accumulation steps: {grad_accum_steps}")

print(f"I am GPU {ddp_rank}")
print("Bye")
import sys; sys.exit(0)

train_loader = DataloaderLite(B=B, T=T)  # * max context/seq len of 1024

torch.set_float32_matmul_precision("high")

model = GPT(
    GPTConfig(vocab_size=50304)
)  # * overriding vocab size to be a *nice* number
# model.eval() # ! only for inference
model.to(device)

# ! significantly reduces run time by decreasing GPU to HBM memory transfers by simulating compilation instead of python interpretation
model = torch.compile(model)

# ! ____________________________________________________________________________________________________

max_lr = 6e-4
min_lr = 0.1 * max_lr  # * 10% of max
warmup_steps = 10  # * steps to get from 0 to max linearly
max_steps = 50


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step > max_steps:
        return min_lr

    else:  # in between -> cos decay ratio
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device
)

for step in range(max_steps):
    t0 = time.time()
    total_loss = 0.0

    optimizer.zero_grad()  #! intialize gradients as 0 -> always do this

    for microstep in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss /= grad_accum_steps  # bc of grad accum, we lose the mean part of mse -> this brings it back
        total_loss += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr = get_lr(step)
    for param in optimizer.param_groups:
        param["lr"] = lr
    torch.cuda.synchronize()  # wait for GPU to finish work before CPU dpes/assigns more work
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tps = tokens_processed // (t1 - t0)
    print(
        f"step {step:4d} | loss: {total_loss.item():.6f} | norm: {norm:.4f} | lr: {lr:.4e} | time: {dt} ms | tokens/s: {tps}"
    )

import sys

sys.exit(0)

""" 
    num_return_sequences = 30
    max_length = 20

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I am a language model, ")
    print(len(tokens))
    tokens = torch.tensor(tokens, dtype=torch.long)
    print(tokens.size())
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    #* generate! right now x is (B, T) where B = 5, T = 8
    #* set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        #* forward the model to get the logits
        with torch.no_grad():
            logits = model(x)  #* (B, T, vocab_size)
            #* take the logits at the last position
            logits = logits[:, -1, :]  #* (B, vocab_size)
            #* get the probabilities
            probs = F.softmax(logits, dim=-1)
            #* do top-k sampling of 50 (huggingface pipeline default)
            #* topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            #* select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1)  #* (B, 1)
            #* gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  #* (B, 1)
            #* append to the sequence
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
"""
