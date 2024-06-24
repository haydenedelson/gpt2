from dataclasses import dataclass
import inspect
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from dataloader import DataLoader
from utils import get_device, get_lr


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
    import os
    from utils import get_device
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    # DDP setup
    # torchrun command sets env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        # use of DDP requires multiple CUDA devices
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = get_device(skip_mps=True)
        print(f"using device {device}")

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('high')

    total_batch_size = 8192 # 2**19, ~0.5M tokens per batch
    B = 4 # micro-batch size
    T = 256 # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total batch size: {total_batch_size}")
        print(f"number of gradient accumulation steps: {grad_accum_steps}")

    # get data loader
    train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

    # load model
    # true vocab_size = 50257, but we increase to 50304 to make it a nice number
    # with no impact on functionality
    model = GPT(GPTConfig(vocab_size=50304))
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"parameters: {num_params}")

    model.eval()
    model = model.to(device)
    if device == 'cuda':
        # when you compile model, torch constructs optimized execution graph
        # reduces read/writes to gpu memory
        model = torch.compile(model)
        print('model compiled')
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # gpt2 original hyperparams
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if device == 'cuda':
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)

            # loss is accumulated (i.e. summed) over grad_accum_steps
            # need to scale down to match original loss
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # clip norm of gradients at 1.0
        # norms should be stable or declining over training
        # if increasing or there is a spike, something is wrong
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0)

        lr = get_lr(step, min_lr, max_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if device == 'cuda':
            torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0)
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    if ddp:
        destroy_process_group()