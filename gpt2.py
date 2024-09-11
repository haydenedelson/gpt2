from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import OmegaConf, DictConfig


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj._RESIDUAL_SCALE_INIT = 1
    
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
        self.c_proj._RESIDUAL_SCALE_INIT = 1
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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # concatenate head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return y


class SelfAttentionBlock(nn.Module):
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
            h = nn.ModuleList([SelfAttentionBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # tie weights of token embedding matrix to weights of lm head
        self.transformer.wte.weight = self.lm_head.weight

        # init params - self.apply will apply to all submodules
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, '_RESIDUAL_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
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
        config = OmegaConf.create(config_args)

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
        # discard masks/buffers from weights
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # get params that require grad
        param_dict = {param_name: val for param_name, val in self.named_parameters() if val.requires_grad}

        # create optim groups
        # 2d parameters will be weight decayed, others will not
        # i.e. biases & layernorms will not be weight decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, num decayed parameters: {num_decay_params:,}")
        print(f"num non-decayed parameter tensors: {len(nondecay_params)}, num non-decayed parameters: {num_nondecay_params:,}")

        # use fused optimizer version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    def forward(self, idx, targets=None):
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

        # Calculate loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss


if __name__ == "__main__":
    import tiktoken
    from utils import get_device

    torch.manual_seed(42)
    device = get_device()

    # get model
    config = OmegaConf.load('config/model/gpt.yaml')
    model = GPT(config)
    model.eval()
    model.to(device)
    
    # get encoder
    encoder = tiktoken.get_encoding('gpt2')

    # encode prefix
    num_return_sequences = 5
    text = "Hello I am GPT. I am a friendly AI."
    tokens = encoder.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    # generate sequences
    x = tokens.to(device)
    max_length = 50
    while x.shape[1] < max_length:
        with torch.no_grad():
            logits, _ = model(x) # (B, T, vocab_size)
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