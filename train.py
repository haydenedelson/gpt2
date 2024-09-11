import hydra
import os
import tiktoken
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

from omegaconf import OmegaConf, DictConfig
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import get_device, get_lr, update_lr
from dataloader import DataLoader
from gpt2 import GPT


def generate_text(model, encoder, num_return_sequences, max_length, device, device_type, ddp_rank):
    tokens = encoder.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = encoder.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")
    

def run_one_epoch(model, data_loader, optimizer, device, train, ddp, grad_accum_steps):
    with torch.set_grad_enabled(train):
        if train:
            optimizer.zero_grad()

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = data_loader.next_batch()
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
            
            if train:
                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

                loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # clip norm of gradients at 1.0
        # norms should be stable or declining over training
        # if increasing or there is a spike, something is wrong
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0)
        return loss_accum, norm


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):
    print(cfg)
    print(cfg.optimizer.params.learning_rate)
    exit()
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
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    total_batch_size = cfg.dataset.total_batch_size # 2**19, ~0.5M tokens per batch
    B = cfg.dataset.micro_batch_size # micro-batch size
    T = cfg.dataset.sequence_len # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total batch size: {total_batch_size}")
        print(f"number of gradient accumulation steps: {grad_accum_steps}")

    # get data loaders
    train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, data_root=cfg.dataset.data_root, split="train")
    val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, data_root=cfg.dataset.data_root, split="val")

    # load model
    # true vocab_size = 50257, but we increase to 50304 to make it a nice number
    # with no impact on functionality
    model = GPT(cfg.model)
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"parameters: {num_params}")

    model = model.to(device)
    if device == 'cuda':
        # when you compile model, torch constructs optimized execution graph
        # reduces read/writes to gpu memory
        model = torch.compile(model)
        print('model compiled')

    # configure optimizer with gpt2 original hyperparams
    optimizer = model.configure_optimizers(device=device, **cfg.optimizer.params)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    model.train()

    # get gpt tokenzier for later generation
    encoder = tiktoken.get_encoding('gpt2')

    max_lr = cfg.scheduler.max_lr
    min_lr = max_lr * cfg.scheduler.min_lr_frac
    warmup_steps = cfg.scheduler.warmup_steps # ~ 375M warmup tokens / 2**19 tokens per step (copying GPT3 paper)
    max_steps = cfg.scheduler.max_steps # ~ 10e9 unique tokens / 2**19 tokens per step
    for step in range(max_steps):
        # validation step
        if step % 100 == 0:
            model.eval()
            val_loader.reset()
            val_loss_accum, val_loss_norm = run_one_epoch(model,
                                                          val_loader,
                                                          optimizer,
                                                          device,
                                                          train=False,
                                                          ddp=ddp,
                                                          grad_accum_steps=20)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
            model.train()
        
        # generate samples
        if step > 0 and step % 100 == 0:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            generate_text(model, encoder, num_return_sequences, max_length, device, device_type, ddp_rank)
            model.train()
        
        # log training time
        t0 = time.time()
        train_loss_accum, train_loss_norm = run_one_epoch(model,
                                                          train_loader,
                                                          optimizer,
                                                          device,
                                                          train=True,
                                                          ddp=ddp,
                                                          grad_accum_steps=grad_accum_steps)
        lr = update_lr(optimizer, step, min_lr, max_lr, warmup_steps, max_steps)
        optimizer.step()

        if device == 'cuda':
            torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0)
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step} | loss: {train_loss_accum.item():.6f} | lr: {lr:.4e} | norm: {train_loss_norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()