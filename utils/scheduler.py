from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, max_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


class WarmupConstantLR(_LRScheduler):
    """Scheduler that does warmup then keeps LR constant."""
    def __init__(self, optimizer: Optimizer, warmup_steps: int, max_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        # After warmup, keep LR constant at base_lr
        return self.base_lrs


def build_scheduler(optimizer: Optimizer, config: Any):
    sched_type = config.get("type", "warmup_cosine")

    if sched_type == "cosine_decay":
        warmup = int(config.get("warmup_steps", 0))
        max_steps = int(config.get("max_steps"))
        min_lr = float(config.get("min_lr", 0.0))
        return WarmupCosineLR(optimizer, warmup, max_steps, min_lr=min_lr)

    if sched_type == "warmup_constant":
        warmup = int(config.get("warmup_steps", 0))
        max_steps = int(config.get("max_steps"))
        return WarmupConstantLR(optimizer, warmup, max_steps)

    if sched_type == "one_cycle":
        max_steps = int(config.get("max_steps"))
        max_lr = float(config.get("max_lr"))
        pct_start = float(config.get("pct_start", 0.3))
        anneal = config.get("anneal_strategy", "cos")
        div_factor = float(config.get("div_factor", 25.0))
        final_div_factor = float(config.get("final_div_factor", 1e4))
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max_steps,
            pct_start=pct_start,
            anneal_strategy=anneal,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    raise ValueError(f"Unknown scheduler type: {sched_type}")