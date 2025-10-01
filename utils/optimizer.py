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

    # Sanity print (enable if you want to verify)
    tot = sum(p.numel() for g in param_groups for p in g["params"])
    print("Param groups:")
    for g in sorted(param_groups, key=lambda d: (d["lr"], d["weight_decay"])):
        n = sum(p.numel() for p in g["params"])
        print(f"  lr={g['lr']:.3g}  wd={g['weight_decay']:.3g}  (#params={n})")
    assert tot == sum(p.numel() for p in model.parameters() if p.requires_grad)
    return opt






# def build_optimizer(model: torch.nn.Module, cfg: dict) -> Optimizer:
#     base_lr = float(cfg["training"]["learning_rate"])
#     wd      = float(cfg["training"].get("weight_decay", 0.0))
#     boost   = 1.25  # ×1.25 for branches + fusion

#     # Name-based routing
#     is_boosted = lambda name: (
#         name.startswith("branches.")                       # all per-branch modules
#         or (".mc_fusion" in name)                          # per-branch GPSFusion
#         # or name.startswith("mm_fusion")                    # GLRFusion
#     )

#     boosted_decay, boosted_no_decay = [], []
#     base_decay,    base_no_decay    = [], []

#     seen = set()
#     for name, p in model.named_parameters():
#         if (not p.requires_grad) or (id(p) in seen):
#             continue
#         seen.add(id(p))

#         # Decay iff it's not a bias and not a norm/affine-1D tensor
#         is_no_decay_tensor = (p.ndim == 1) or name.endswith(".bias")

#         if is_boosted(name):
#             (boosted_no_decay if is_no_decay_tensor else boosted_decay).append(p)
#         else:
#             (base_no_decay if is_no_decay_tensor else base_decay).append(p)

#     # Sanity: avoid empty groups (harmless, but tidy)
#     param_groups = []
#     if boosted_decay:
#         param_groups.append({"params": boosted_decay, "lr": base_lr * boost, "weight_decay": wd})
#     if boosted_no_decay:
#         param_groups.append({"params": boosted_no_decay, "lr": base_lr * boost, "weight_decay": 0.0})
#     if base_decay:
#         param_groups.append({"params": base_decay, "lr": base_lr, "weight_decay": wd})
#     if base_no_decay:
#         param_groups.append({"params": base_no_decay, "lr": base_lr, "weight_decay": 0.0})

#     # You can pass betas/eps from cfg if you like
#     optimizer = torch.optim.AdamW(param_groups)  # , betas=(0.9,0.999), eps=1e-8

#     # Optional: quick print to verify sizes (comment in if debugging)
#     tot = sum(p.numel() for g in param_groups for p in g["params"])
#     print("boosted_decay:", sum(p.numel() for p in boosted_decay))
#     print("boosted_no_decay:", sum(p.numel() for p in boosted_no_decay))
#     print("base_decay:", sum(p.numel() for p in base_decay))
#     print("base_no_decay:", sum(p.numel() for p in base_no_decay))
#     assert tot == sum(p.numel() for p in model.parameters() if p.requires_grad)

#     return optimizer




















# def build_optimizer(model: torch.nn.Module, cfg: dict) -> Optimizer:
#     lr = cfg["training"]["learning_rate"]
#     wd = cfg["training"].get("weight_decay", 0.0)

#     decay, no_decay = [], []

#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         if p.ndim == 1 or name.endswith(".bias"):
#             no_decay.append(p)
#         else:
#             decay.append(p)

#     optimizer = torch.optim.AdamW(
#         [
#             {"params": decay, "weight_decay": wd},
#             {"params": no_decay, "weight_decay": 0.0},
#         ],
#         lr=lr,
#     )

#     return optimizer






# def build_optimizer(model: torch.nn.Module, cfg: dict) -> Optimizer:
#     lr = cfg["training"]["learning_rate"]
#     wd = cfg["training"].get("weight_decay", 0.0)
#     decay, no_decay, classifier_decay, classifier_no_decay = [], [], [], []

#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         if p.ndim == 1 or name.endswith(".bias"):
#             if "classifier" in name:
#                 classifier_no_decay.append(p)
#             else:
#                 no_decay.append(p)
#         else:
#             if "classifier" in name:
#                 classifier_decay.append(p)
#             else:
#                 decay.append(p)
            
#     optimizer = torch.optim.AdamW(
#         [{"params": decay, "weight_decay": wd},
#          {"params": no_decay, "weight_decay": 0.0},
#          {"params": classifier_decay, "weight_decay": 2.0 * wd},
#          {"params": classifier_no_decay, "weight_decay": 0.0}],
#         lr=lr)
    
#     return optimizer







# def build_optimizer(model: torch.nn.Module, cfg: dict) -> Optimizer:
#     training_cfg = cfg["training"]
#     lr = training_cfg["learning_rate"]
#     wd = training_cfg.get("weight_decay", 0.0)

#     # scale classifier LR to keep backbone updates dominant early on
#     cls_scale = float(training_cfg.get("classifier_lr_scale", 1.0))
#     cls_scale = max(0.0, cls_scale)

#     param_buckets = {
#         ("backbone", "decay"): [],
#         ("backbone", "no_decay"): [],
#         ("classifier", "decay"): [],
#         ("classifier", "no_decay"): [],
#     }

#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         group = "classifier" if "classifier" in name else "backbone"
#         decay_key = "no_decay" if (p.ndim == 1 or name.endswith(".bias")) else "decay"
#         param_buckets[(group, decay_key)].append(p)

#     param_groups = []
#     for group_name in ("backbone", "classifier"):
#         for decay_key in ("decay", "no_decay"):
#             params = param_buckets[(group_name, decay_key)]
#             if not params:
#                 continue

#             is_decay = decay_key == "decay"
#             group_lr = lr * (cls_scale if group_name == "classifier" else 1.0)
#             group_cfg = {
#                 "params": params,
#                 "weight_decay": wd if is_decay else 0.0,
#                 "lr": group_lr,
#                 "tag": group_name,
#             }

#             if group_name == "classifier":
#                 group_cfg["target_lr"] = lr
#                 group_cfg["lr_scale"] = cls_scale

#             param_groups.append(group_cfg)

#     optimizer = torch.optim.AdamW(param_groups, lr=lr)

#     return optimizer
