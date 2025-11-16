from __future__ import annotations

import torch
from typing import Dict

import torch
from torch.optim import Optimizer


def build_optimizer(model: torch.nn.Module, cfg: Dict) -> Optimizer:
    base_lr = float(cfg["training"]["learning_rate"])
    wd      = float(cfg["training"].get("weight_decay", 0.0))

    LR_SCALES = {
        "branches": 1.25, 
        "mm_fusion": 1.25, 
        "individual_classifier.classifiers": 0.50,  
        "group_classifier.classifiers": 0.50,  
        "individual_classifier.adapters": 1.50,
        "group_classifier.adapters": 1.50,  
    }

    # no decay
    PROTO_KEYS = ("prototype", "prototypes", "proto", "centroid", "centroids",
                  "class_means", "classmeans", "memory_bank", "weight_bank")

    # no decay
    TEMP_KEYS = ("temperature", "logit_scale", "scale", "tau", "phi", "beta")

    def lr_scale_for(name: str) -> float:
        if name.startswith("branches."):
            return LR_SCALES["branches"]
        if name.startswith("mm_fusion"):
            return LR_SCALES["mm_fusion"]
        if name.startswith("individual_classifier.classifiers"):
            return LR_SCALES["individual_classifier.classifiers"]
        if name.startswith("group_classifier.classifiers"):
            return LR_SCALES["group_classifier.classifiers"]
        if name.startswith("individual_classifier.adapters"):
            return LR_SCALES["individual_classifier.adapters"]
        if name.startswith("group_classifier.adapters"):
            return LR_SCALES["group_classifier.adapters"]
        return 1.0

    def no_decay_special(name: str, p: torch.nn.Parameter) -> bool:
        lname = name.lower()
        if p.ndim == 1 or name.endswith(".bias"):
            return True
        if any(k in lname for k in PROTO_KEYS):
            return True
        if any(lname.endswith(f".{k}") or f".{k}." in lname for k in TEMP_KEYS):
            return True
        if lname.startswith("individual_classifier.adapters"):
            return True
        if lname.startswith("group_classifier.adapters"):
            return True
        return False

    # Build param groups keyed by (lr, weight_decay)
    groups: Dict[tuple, list] = {}
    seen = set()

    for name, p in model.named_parameters():
        if (not p.requires_grad) or (id(p) in seen):
            continue
        seen.add(id(p))

        lr = base_lr * lr_scale_for(name)
        wd_here = 0.0 if no_decay_special(name, p) else wd
        key = (lr, wd_here)
        groups.setdefault(key, []).append(p)

    # Turn into optimizer param_groups
    param_groups = [{"params": ps, "lr": lr, "weight_decay": wd_here}
                    for (lr, wd_here), ps in groups.items() if ps]

    opt = torch.optim.AdamW(param_groups)

    # # Sanity print
    # tot = sum(p.numel() for g in param_groups for p in g["params"])
    # print("Param groups:")
    # for g in sorted(param_groups, key=lambda d: (d["lr"], d["weight_decay"])):
    #     n = sum(p.numel() for p in g["params"])
    #     print(f"  lr={g['lr']:.3g}  wd={g['weight_decay']:.3g}  (#params={n})")
    # assert tot == sum(p.numel() for p in model.parameters() if p.requires_grad)

    return opt
