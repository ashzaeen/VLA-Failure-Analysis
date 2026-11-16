import numpy as np
import torch
def _to_numpy(x):
    """Convert torch or numpy to numpy"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _to_float32(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    return x.astype(np.float32)


def _mean(x, axis):
    if isinstance(x, torch.Tensor):
        return x.mean(dim=axis)
    return x.mean(axis=axis)