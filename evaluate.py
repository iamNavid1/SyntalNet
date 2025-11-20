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
    p.add_argument("--checkpoint", required=True, type=str, 
                   help="Path to checkpoint file or directory. For kfold/logo, can be directory with fold subdirs.")
    p.add_argument("--out", type=str, default=None, help="Optional path to write metrics JSON")
    p.add_argument("--fold-idx", type=int, default=None,
                   help="Optional: evaluate specific fold (for kfold/logo modes). If not specified, evaluates all folds.")
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
    g = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(n, generator=g).tolist()
    
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


def build_val_datasets(cfg: dict, fold_idx: int = None) -> List[Tuple[Subset | GroupDynamicsDataset, int | None]]:
    """
    Build validation datasets based on the split mode specified in the config ("item", "group", "logo", "kfold").
    Args:
        cfg: Configuration dictionary with dataset settings
        fold_idx: Optional specific fold to evaluate (for kfold/logo). If None, returns all folds.
    Returns:
        List of (val_subset, fold_identifier) tuples.
        - For item/group: single tuple with fold_identifier=None
        - For kfold: tuples with fold_identifier=fold_index (0 to n_folds-1)
        - For logo: tuples with fold_identifier=group_id
    """
    args = dict(cfg["dataset"].get("args", {}))
    
    # Load normalization stats
    norm_stats = cfg["dataset"].get("stats_dir")
    if isinstance(norm_stats, str):
        with open(norm_stats, "r") as f:
            norm_stats = yaml.safe_load(f)
    
    if norm_stats:
        transform = StandardizeTransform(norm_stats)
        args["transforms"] = transform
    
    # Get split configuration
    split_cfg = cfg["dataset"].get("split", {"mode": "item", "ratio": 0.2})
    mode = split_cfg.get("mode", "item")
    split_seed = int(split_cfg.get("seed", cfg.get("training", {}).get("seed", 42)))
    
    folds = []
    
    if mode == "item":
        # Item-based split: single validation set
        full = GroupDynamicsDataset(**args)
        n = len(full)
        n_val = int(round(n * float(split_cfg.get("ratio", 0.2))))
        n_train = n - n_val
        g = torch.Generator().manual_seed(split_seed)
        _, val_subset = random_split(full, [n_train, n_val], generator=g)
        folds.append((val_subset, None))
    
    elif mode == "group":
        # Group-based split: single validation set with specified groups
        all_groups = discover_group_ids(args["root_dir"], args["modalities"])
        val_groups = split_cfg.get("groups")
        
        if not val_groups:
            rng = random.Random(split_seed)
            k = max(1, int(round(len(all_groups) * float(split_cfg.get("ratio", 0.2)))))
            val_groups = sorted(rng.sample(all_groups, k))
        
        val_cfg = dict(args)
        val_cfg["include_groups"] = val_groups
        val_dataset = GroupDynamicsDataset(**val_cfg)
        folds.append((val_dataset, None))
    
    elif mode == "logo":
        # Leave-one-group-out: one fold per group
        all_groups = discover_group_ids(args["root_dir"], args["modalities"])
        
        if fold_idx is not None:
            # Evaluate specific fold
            if fold_idx < len(all_groups):
                held_out_group = all_groups[fold_idx]
                val_cfg = dict(args)
                val_cfg["include_groups"] = [held_out_group]
                val_dataset = GroupDynamicsDataset(**val_cfg)
                folds.append((val_dataset, held_out_group))
        else:
            # Evaluate all folds
            for held_out_group in all_groups:
                val_cfg = dict(args)
                val_cfg["include_groups"] = [held_out_group]
                val_dataset = GroupDynamicsDataset(**val_cfg)
                folds.append((val_dataset, held_out_group))
    
    elif mode == "kfold":
        # K-fold cross-validation: all folds or specific fold
        n_folds = int(split_cfg.get("n_folds", 5))
        full_dataset = GroupDynamicsDataset(**args)
        
        if fold_idx is not None:
            # Evaluate specific fold
            if 0 <= fold_idx < n_folds:
                _, val_subset = _create_kfold_splits(full_dataset, n_folds, fold_idx, split_seed)
                folds.append((val_subset, fold_idx))
        else:
            # Evaluate all folds
            for f_idx in range(n_folds):
                _, val_subset = _create_kfold_splits(full_dataset, n_folds, f_idx, split_seed)
                folds.append((val_subset, f_idx))
    
    else:
        raise ValueError(f"Unknown split.mode: {mode}")
    
    return folds


def _get_latest_epoch_file(epoch_files: List[str]) -> str:
    def extract_epoch_num(filepath: str) -> int:
        basename = os.path.basename(filepath)
        match = re.match(r"epoch_(\d+)\.pth$", basename)
        if match:
            return int(match.group(1))
        return -1  # Invalid format, sort to beginning
    # Sort by epoch number (numeric)
    sorted_files = sorted(epoch_files, key=extract_epoch_num)
    return sorted_files[-1]  # Latest epoch


def find_checkpoint_path(checkpoint_path: str, split_mode: str, fold_idx: int = None, held_out_group: int = None) -> str:
    """ Find checkpoint path based on split mode and fold. """
    if os.path.isfile(checkpoint_path):
        # Direct file path
        return checkpoint_path
    
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    
    # Directory path - need to find the right checkpoint
    if split_mode in ("item", "group"):
        # Single checkpoint in directory
        for name in ["best.pth", "latest.pth", "checkpoint.pth"]:
            path = os.path.join(checkpoint_path, name)
            if os.path.isfile(path):
                return path
        # Try epoch files
        epoch_files = glob.glob(os.path.join(checkpoint_path, "epoch_*.pth"))
        if epoch_files:
            return _get_latest_epoch_file(epoch_files)
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    
    elif split_mode == "kfold":
        # Checkpoint in fold subdirectory
        if fold_idx is None:
            raise ValueError("fold_idx required for kfold mode")
        fold_dir = os.path.join(checkpoint_path, "kfold", f"fold_{fold_idx:02d}")
        for name in ["best.pth", "latest.pth", "checkpoint.pth"]:
            path = os.path.join(fold_dir, name)
            if os.path.isfile(path):
                return path
        # Try epoch files
        epoch_files = glob.glob(os.path.join(fold_dir, "epoch_*.pth"))
        if epoch_files:
            return _get_latest_epoch_file(epoch_files)
        raise FileNotFoundError(f"No checkpoint found in {fold_dir}")
    
    elif split_mode == "logo":
        # Checkpoint in fold subdirectory (by group ID)
        if held_out_group is None:
            raise ValueError("held_out_group required for logo mode")
        fold_dir = os.path.join(checkpoint_path, "logo", f"fold_{held_out_group:02d}")
        for name in ["best.pth", "latest.pth", "checkpoint.pth"]:
            path = os.path.join(fold_dir, name)
            if os.path.isfile(path):
                return path
        # Try epoch files
        epoch_files = glob.glob(os.path.join(fold_dir, "epoch_*.pth"))
        if epoch_files:
            return _get_latest_epoch_file(epoch_files)
        raise FileNotFoundError(f"No checkpoint found in {fold_dir}")
    
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")


def aggregate_metrics(all_metrics: List[dict]) -> dict:
    """ Aggregate metrics across multiple folds."""
    if not all_metrics:
        return {}
    
    if len(all_metrics) == 1:
        return all_metrics[0]
    
    # Collect all metric keys
    def collect_keys(d, prefix=""):
        keys = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(collect_keys(v, full_key))
            else:
                keys.append(full_key)
        return keys
    
    all_keys = set()
    for m in all_metrics:
        all_keys.update(collect_keys(m))
    
    # Aggregate metrics
    aggregated = {}
    
    def get_nested_value(d, key_path):
        keys = key_path.split(".")
        val = d
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return None
        return val
    
    def set_nested_value(d, key_path, value):
        keys = key_path.split(".")
        current = d
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    for key in sorted(all_keys):
        values = []
        for m in all_metrics:
            val = get_nested_value(m, key)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            median_val = np.median(values)
            q1_val = np.percentile(values, 25)
            q3_val = np.percentile(values, 75)

            set_nested_value(aggregated, key, {
                "mean": float(mean_val),
                "std": float(std_val),
                "min": float(min_val),
                "max": float(max_val),
                "median": float(median_val),
                "q1": float(q1_val),
                "q3": float(q3_val),
                "n_folds": len(values)
            })
    
    return aggregated


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

    # Get split mode
    split_cfg = cfg["dataset"].get("split", {"mode": "item", "ratio": 0.2})
    split_mode = split_cfg.get("mode", "item")
    
    # Build validation datasets (all folds or specific fold)
    print(f"Building validation datasets (split mode: {split_mode})...")
    val_folds = build_val_datasets(cfg, fold_idx=args.fold_idx)
    n_folds = len(val_folds)
    print(f"Created {n_folds} fold(s) for evaluation")
    
    # DataLoader settings
    val_bs = int(cfg["training"].get("val_batch_size", cfg["training"].get("batch_size", 8)))
    num_workers = int(cfg["training"].get("num_workers", 4))
    prefetch_factor = cfg["training"].get("prefetch_factor", 4)
    pin_memory_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else ""
    
    # Validation setup
    autocast_kwargs = BuildAutocastKWargs(cfg, device)
    validator = Validator(device=device, autocast_kwargs=autocast_kwargs)
    
    all_fold_metrics = []
    
    # Evaluate each fold
    for fold_num, (val_dataset, fold_identifier) in enumerate(val_folds):
        fold_label = f"fold {fold_identifier}" if fold_identifier is not None else "validation set"
        if n_folds > 1:
            print(f"\nEvaluating {fold_label} ({fold_num + 1}/{n_folds})...")
        else:
            print("Building Dataloader...")
        
        # Build data loader for this fold
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
        
        # Load model checkpoint for this fold
        print("Building Model...")
        model = build.model(cfg).to(device)
        
        # Find checkpoint path for this fold
        if n_folds > 1 and split_mode in ("kfold", "logo"):
            # For multi-fold evaluation, try to find fold-specific checkpoint
            try:
                checkpoint_path = find_checkpoint_path(
                    args.checkpoint,
                    split_mode,
                    fold_idx=fold_identifier if split_mode == "kfold" else None,
                    held_out_group=fold_identifier if split_mode == "logo" else None
                )
                print(f"Loading checkpoint: {checkpoint_path}")
            except (FileNotFoundError, ValueError):
                # Fall back to provided checkpoint if fold-specific not found
                checkpoint_path = args.checkpoint
                if n_folds > 1:
                    print(f"Warning: Fold-specific checkpoint not found, using: {checkpoint_path}")
        else:
            checkpoint_path = args.checkpoint
        
        state = torch.load(checkpoint_path, map_location="cpu")
        print("Loading Model Checkpoints...")
        
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if missing or unexpected:
            print(f"[LD] missing: {missing}")
            print(f"[LD] unexpected: {unexpected}")
        assert len(missing) == 0 and len(unexpected) == 0, "State mismatch!"
        model.eval()
        
        # Run evaluation
        if n_folds > 1:
            print(f"Evaluation in Progress for {fold_label}...")
        else:
            print("Evaluation in Progress...")
        
        with torch.inference_mode():
            metrics, _ = validator.run(model, loader)
            all_fold_metrics.append(metrics)
        
        # Cleanup
        del loader
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Aggregate results
    if n_folds > 1:
        print("\nAggregating results across folds...")
        aggregated_metrics = aggregate_metrics(all_fold_metrics)
        payload = {
            "per_fold": [round_jsonable(m, ndigits=6) for m in all_fold_metrics],
            "aggregated": round_jsonable(aggregated_metrics, ndigits=6),
            "n_folds": n_folds,
            "split_mode": split_mode
        }
    else:
        payload = round_jsonable(all_fold_metrics[0], ndigits=6)
    
    # Save or print results
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[OK] Saved metrics to {args.out}")
    else:
        print("\n" + "=" * 80)
        print("Evaluation Results:")
        print("=" * 80)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
