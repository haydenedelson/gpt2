from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # key, query, & value projection matrices
        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed)
        # output projection
        self.c_proj = nn.Linear(self.n_embed, self.n_embed)
        # causal mask
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)
                                                .view(1, 1, config.block_size, config.block_size)))
    
    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimension

        # query, key, value projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)

        # nh = number of heads, hs = head size, C (embedding dim) = nh * hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # multi-head self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        # concatenate head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weights"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # n_layer, n_head, & n_embed are determined from model_type
        base_configs = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600)
        }

        # get config
        config_args = base_configs[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)

        # get model
        model = GPT(config)
        state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        state_dict_keys = [k for k in state_dict_keys if not k.endswith('.attn.bias')] # discard this mask/buffer from weights

        # get huggingface model to get pretrained weights
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        state_dict_hf = model_hf.state_dict()

        # copy weights from hf model
        state_dict_hf_keys = state_dict_hf.keys()
        state_dict_hf_keys = [k for k in state_dict_hf_keys if not k.endswith('.attn.masked_bias')]
        state_dict_hf_keys = [k for k in state_dict_hf_keys if not k.endswith('.attn.bias')]
        # some weights are transposed, need to fix manually
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # assert that state dicts are the same length
        assert len(state_dict_keys) == len(state_dict_hf_keys), f"keys mismatched: {len(state_dict_keys)} != {len(state_dict_hf_keys)}"

        # fix transposed keys
        for k in state_dict_hf_keys:
            if any(k.endswith(suffix) for suffix in transposed):
                # assert that matrices have same shape, but transposed
                assert state_dict_hf[k].shape[::-1] == state_dict[k].shape
                # copy from huggingface state dict to target state dict
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k].transpose(0, 1))
            else:
                assert state_dict_hf[k].shape == state_dict[k].shape
                # copy from huggingface state dict to target state dict
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k])
        
        return model
    
    def forward(self, idx):
        # input is token indices
        # idx is of shape (B, T)
        B, T = idx.shape

        # assert input is not larger than max sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        # forward the token & position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embed)
        x = tok_emb + pos_emb # position embeddings are broadcasted to add with token embeddings

        # forward through blocks of transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward through final layernorm & classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits


if __name__ == "__main__":
    import tiktoken
    from utils import get_device

    device = get_device()
    print(f"using device {device}")

    # load model
    model = GPT.from_pretrained('gpt2')
    print('weights loaded')
    model.eval()
    model = model.to(device)

    # get encoder
    encoder = tiktoken.get_encoding('gpt2')
    
    # encode prefix
    prefix = "Hello, I'm a language model,"
    num_return_sequences = 5
    tokens = encoder.encode(prefix)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    # generate sequences
    torch.manual_seed(42)
    max_length = 30
    while x.shape[1] < max_length:
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # do top-k sampling of 50 (huggingface pipeline default)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # sample 1 index from topk_probs
            next_idx = torch.multinomial(topk_probs, 1)
            # gather corresponding indices
            next_tok_idx = torch.gather(topk_indices, 1, next_idx)
            # append to the sequence
            x = torch.cat([x, next_tok_idx], dim=1)
    
    for i in range(num_return_sequences):
        tokens = x[i].tolist()
        decoded = encoder.decode(tokens)
        print(">", decoded)

