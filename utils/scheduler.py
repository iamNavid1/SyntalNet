from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, max_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]
        step = self.last_epoch + 1
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


def build_scheduler(optimizer: Optimizer, config: Any) -> WarmupCosineLR:
    warmup = int(config.get("warmup_steps", 0))
    max_steps = int(config.get("max_steps"))
    min_lr = float(config.get("min_lr", 0.0))
    return WarmupCosineLR(optimizer, warmup, max_steps, min_lr=min_lr)
