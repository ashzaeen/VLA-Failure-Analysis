from __future__ import annotations

import random

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
