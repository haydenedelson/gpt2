import os
import time
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import get_device, get_lr
from dataloader import DataLoader
from gpt2 import GPT, GPTConfig


def main():
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

    # gpt2 original hyperparams
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

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


if __name__ == "__main__":
    main()