from __future__ import annotations

import argparse
import json
import yaml
import os
import random
import re
import glob
from typing import List

import torch
from torch.utils.data import DataLoader, random_split, Subset

from data.dataset import GroupDynamicsDataset
from data.collate import collate_fn
from data.transforms import StandardizeTransform
from engine.validator import Validator
from engine.utils import BuildAutocastKWargs
from models.builder import build_model, load_config


# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--out", type=str, default=None, help="Optional path to write metrics JSON")
    return p.parse_args()


# ----------------------------- Dataset helpers (mirror training logic) -----------------------------

def discover_group_ids(root_dir: str, modalities: List[str]) -> List[int]:
    example_mod = modalities[0]
    pattern_csv = os.path.join(root_dir, example_mod, "Group_*.csv")
    pattern_json = os.path.join(root_dir, example_mod, "Group_*.json")
    files = glob.glob(pattern_csv) + glob.glob(pattern_json)
    group_ids = set()
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"Group_(\d+)\.(csv|json)$", base)
        if m:
            group_ids.add(int(m.group(1)))
    return sorted(group_ids)


def build_val_dataset(cfg: dict) -> GroupDynamicsDataset | Subset:
    args = dict(cfg["dataset"]["args"])
    norm_stats = cfg["dataset"].get("stats_dir")
    if isinstance(norm_stats, str):
        with open(norm_stats, "r") as f:
            norm_stats = yaml.safe_load(f)
    if norm_stats:
        transform = StandardizeTransform(norm_stats)
        args["transforms"] = transform

    split_cfg = cfg["dataset"].get("split", {"mode": "item", "ratio": 0.2})
    mode = split_cfg.get("mode", "item")
    split_seed = int(split_cfg.get("seed", cfg["training"].get("seed", 42)))

    if mode == "item":
        # Build full dataset then carve out validation subset with the same seed logic used in training
        full = GroupDynamicsDataset(**args)
        n = len(full)
        n_val = int(round(n * float(split_cfg.get("ratio", 0.2))))
        n_train = n - n_val
        g = torch.Generator().manual_seed(split_seed)
        _, val_subset = random_split(full, [n_train, n_val], generator=g)
        return val_subset

    elif mode == "group":
        # Use explicit group list if provided, otherwise reproduce deterministic sampling from seed
        all_groups = discover_group_ids(args["root_dir"], args["modalities"])
        val_groups = split_cfg.get("groups")
        if not val_groups:
            rng = random.Random(split_seed)
            k = max(1, int(round(len(all_groups) * float(split_cfg.get("ratio", 0.2)))))
            val_groups = sorted(rng.sample(all_groups, k))
        val_args = dict(args)
        val_args["include_groups"] = val_groups
        return GroupDynamicsDataset(**val_args)

    else:
        raise ValueError(f"Unknown split.mode: {mode}")


# ----------------------------- JSON utils -----------------------------

def round_jsonable(obj, ndigits: int = 6):
    """Convert metrics dict (with numpy/tensors) to JSON-able, rounded Python types."""
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: round_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_jsonable(v, ndigits) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in {"f"}:
            return np.round(obj, ndigits).tolist()
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), ndigits)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        x = obj.detach().cpu().numpy()
        return round_jsonable(x, ndigits)
    if isinstance(obj, set):
        return [round_jsonable(v, ndigits) for v in sorted(obj)]
    return obj


# ----------------------------- Main -----------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    val_dataset = build_val_dataset(cfg)
    val_bs = int(cfg["training"].get("val_batch_size", cfg["training"].get("batch_size", 8)))
    num_workers = int(cfg["training"].get("num_workers", 4))

    loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
    )

    # Model
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    # Validation
    autocast_kwargs = BuildAutocastKWargs(cfg, device)
    validator = Validator(device=device, autocast_kwargs=autocast_kwargs)

    metrics, _ = validator.run(model, loader)

    payload = round_jsonable(metrics, ndigits=6)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] Saved metrics to {args.out}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
