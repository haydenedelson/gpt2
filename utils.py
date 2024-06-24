import math
import torch


def get_device(skip_mps=False):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif not skip_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    return device


def get_lr(step, min_lr, max_lr, warmup_steps, max_steps):
    # linear warmup for warmup_steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # if step > max_step, return min learning rate
    if step > max_steps:
        return min_lr

    # in between, use cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

