from __future__ import annotations

import argparse
import json
import yaml
import os
import random
import re
import glob
import numpy as np
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split, Subset

from data.dataset import GroupDynamicsDataset
from data.collate import collate_fn
from data.transforms import StandardizeTransform
from engine.validator import Validator
from engine.utils import BuildAutocastKWargs
import models.builders as build


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--out", type=str, default=None, help="Optional path to write metrics JSON")
    return p.parse_args()


# ----------------------------- distributed utils -----------------------------
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode() -> Tuple[bool, int, int, int, torch.device]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = env_world_size > 1

    if distributed:
        rank = int(os.environ["RANK"])
        world_size = env_world_size
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else None

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    return distributed, rank, world_size, local_rank, device


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


# ----------------------------- reproducibility -----------------------------
def set_seed(seed: int, add_rank: bool = True):
    s = seed + get_rank() if (add_rank and is_dist_avail_and_initialized()) else seed
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


# ----------------------------- Dataset helpers -----------------------------
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


def _create_kfold_splits(dataset, n_folds: int, fold_idx: int, seed: int):
    """Create train/val splits for a specific fold in K-fold CV."""
    n = len(dataset)
    
    # Shuffle indices
    rng = random.Random(seed)
    all_indices = list(range(n))
    rng.shuffle(all_indices)
    
    # Calculate fold boundaries
    fold_size = n // n_folds
    remainder = n % n_folds
    start_idx = fold_idx * fold_size + min(fold_idx, remainder)
    end_idx = start_idx + fold_size + (1 if fold_idx < remainder else 0)
    
    # Split indices
    val_indices = all_indices[start_idx:end_idx]
    train_indices = all_indices[:start_idx] + all_indices[end_idx:]
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    return train_subset, val_subset

def build_val_dataset(cfg: dict) -> GroupDynamicsDataset | Subset:
    args = dict(cfg["dataset"].get("args"))
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
        full = GroupDynamicsDataset(**args)
        n = len(full)
        n_val = int(round(n * float(split_cfg.get("ratio", 0.2))))
        n_train = n - n_val
        g = torch.Generator().manual_seed(split_seed)
        _, val_subset = random_split(full, [n_train, n_val], generator=g)
        return val_subset

    elif mode == "group" or mode == "logo":
        all_groups = discover_group_ids(args["root_dir"], args["modalities"])
        val_groups = split_cfg.get("groups")

        if not val_groups:
            rng = random.Random(split_seed)
            k = max(1, int(round(len(all_groups) * float(split_cfg.get("ratio", 0.2)))))
            val_groups = sorted(rng.sample(all_groups, k))

        val_args = dict(args)
        val_args["include_groups"] = val_groups
        return GroupDynamicsDataset(**val_args)

    elif mode == "kfold":
        n_folds = int(split_cfg.get("n_folds", 5))
        fold_idx = split_cfg.get("fold_idx")

        if not fold_idx:
            rng = random.Random(split_seed)
            fold_idx = rng.randint(0, n_folds - 1)

        full = GroupDynamicsDataset(**args)
        _, val_subset = _create_kfold_splits(full, n_folds, fold_idx, split_seed)
        return val_subset

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
    cfg = build.config(args.config)

    set_seed(int(cfg.get("training", {}).get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- Data --------------------
    print("Building Dataloader...")
    val_dataset = build_val_dataset(cfg)

    val_bs = int(cfg["training"].get("val_batch_size", cfg["training"].get("batch_size", 8)))
    num_workers = int(cfg["training"].get("num_workers", 4))
    prefetch_factor = cfg["training"].get("prefetch_factor", 4)
    pin_memory_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else ""

    loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(1 if not is_dist_avail_and_initialized() else get_rank() + 1),
        pin_memory_device=pin_memory_device,
    )


    # -------------------- Model --------------------
    print("Building Model...")
    model = build.model(cfg).to(device)

    state = torch.load(args.checkpoint, map_location="cpu")
    print("Loading Model Checkpoints...")

    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print("[LD] missing:", missing)
    print("[LD] unexpected:", unexpected)
    assert len(missing) == 0 and len(unexpected) == 0, "State mismatch!"
    model.eval()


    # -------------------- Validation --------------------
    print("Evaluation in Progress...")
    autocast_kwargs = BuildAutocastKWargs(cfg, device)
    validator = Validator(device=device, autocast_kwargs=autocast_kwargs)

    with torch.inference_mode():
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
