from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer


def build_optimizer(model: torch.nn.Module, cfg: dict) -> Optimizer:
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"].get("weight_decay", 0.0)
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
            
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr)
    
    return optimizer
