from __future__ import annotations
from typing import List, Tuple, Optional
import os
import sys
import re
import glob
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import yaml

from data.dataset import GroupDynamicsDataset
from data.collate import collate_fn
from data.transforms import StandardizeTransform

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def discover_group_ids(root_dir: str, modalities: List[str]) -> List[int]:
    """
    Discover all group IDs from the dataset directory.
    
    Args:
        root_dir: Root directory containing modality subdirectories
        modalities: List of modality names
    
    Returns:
        Sorted list of group IDs
    """
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


def _create_kfold_splits(
    dataset: GroupDynamicsDataset,
    n_folds: int,
    fold_idx: int,
    seed: int
) -> Tuple[Subset, Subset]:
    """
    Create train/val splits for a specific fold in K-fold CV.
    
    Args:
        dataset: Full dataset
        n_folds: Number of folds
        fold_idx: Index of fold to create (0 to n_folds-1)
        seed: Random seed for shuffling
    
    Returns:
        (train_subset, val_subset)
    """
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


def build_datasets(cfg: dict) -> List[Tuple[Subset | GroupDynamicsDataset, Subset | GroupDynamicsDataset]]:
    """
    Build datasets based on the split mode specified in the config.
    
    Args:
        cfg: Configuration dictionary with dataset settings
    
    Returns:
        List of (train_subset, val_subset) tuples.
        For kfold mode: returns all folds.
        For other modes: returns a single fold (validation set only, train_subset is None).
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
        train_subset, val_subset = random_split(full, [n_train, n_val], generator=g)
        folds.append((train_subset, val_subset))
    
    elif mode == "group":
        # Group-based split: single validation set with specified groups
        all_groups = discover_group_ids(args["root_dir"], args["modalities"])
        val_groups = split_cfg.get("groups")
        
        if not val_groups:
            rng = random.Random(split_seed)
            k = max(1, int(round(len(all_groups) * float(split_cfg.get("ratio", 0.2)))))
            val_groups = sorted(rng.sample(all_groups, k))
        
        train_cfg = dict(args)
        val_cfg = dict(args)
        train_cfg["exclude_groups"] = val_groups
        val_cfg["include_groups"] = val_groups
        
        train_dataset = GroupDynamicsDataset(**train_cfg)
        val_dataset = GroupDynamicsDataset(**val_cfg)
        folds.append((train_dataset, val_dataset))
    
    elif mode == "logo":
        # Leave-one-group-out: one fold per group
        all_groups = discover_group_ids(args["root_dir"], args["modalities"])
        
        for held_out_group in all_groups:
            train_cfg = dict(args)
            val_cfg = dict(args)
            train_cfg["exclude_groups"] = [held_out_group]
            val_cfg["include_groups"] = [held_out_group]
            
            train_dataset = GroupDynamicsDataset(**train_cfg)
            val_dataset = GroupDynamicsDataset(**val_cfg)
            folds.append((train_dataset, val_dataset))
    
    elif mode == "kfold":
        # K-fold cross-validation: all folds
        n_folds = int(split_cfg.get("n_folds", 5))
        full_dataset = GroupDynamicsDataset(**args)
        
        for fold_idx in range(n_folds):
            train_subset, val_subset = _create_kfold_splits(
                full_dataset, n_folds, fold_idx, split_seed
            )
            folds.append((train_subset, val_subset))
    
    else:
        raise ValueError(f"Unknown split.mode: {mode}")
    
    return folds


# Backward compatibility alias
def build_kfold_datasets(
    cfg: dict,
    n_folds: int = 5,
    seed: Optional[int] = None
) -> List[Tuple[Subset, Subset]]:
    """
    Build K-fold CV datasets (backward compatibility).
        
    Args:
        cfg: Configuration dictionary with dataset settings
        n_folds: Number of folds (ignored if config specifies n_folds)
        seed: Random seed (ignored if config specifies seed)
    
    Returns:
        List of (train_subset, val_subset) tuples, one per fold
    """
    # Make a deep copy to avoid modifying the original config
    import copy
    cfg_copy = copy.deepcopy(cfg)
    
    # Override config if needed for backward compatibility
    if "split" not in cfg_copy.get("dataset", {}):
        cfg_copy["dataset"]["split"] = {"mode": "kfold", "n_folds": n_folds}
    elif cfg_copy["dataset"]["split"].get("mode") != "kfold":
        # If mode is not kfold, temporarily override it
        cfg_copy["dataset"]["split"]["mode"] = "kfold"
        cfg_copy["dataset"]["split"]["n_folds"] = n_folds
        if seed is not None:
            cfg_copy["dataset"]["split"]["seed"] = seed
    
    return build_datasets(cfg_copy)


def build_dataloader(
    dataset: Subset | GroupDynamicsDataset,
    cfg: dict,
    shuffle: bool = False,
    is_val: bool = True
) -> DataLoader:
    """
    Build a DataLoader for a dataset.
    
    Args:
        dataset: Dataset or subset to load
        cfg: Configuration dictionary
        shuffle: Whether to shuffle
        is_val: Whether this is a validation loader (affects batch size)
    
    Returns:
        DataLoader
    """
    training_cfg = cfg.get("training", {})
    
    if is_val:
        batch_size = int(training_cfg.get("val_batch_size", training_cfg.get("batch_size", 12)))
    else:
        batch_size = int(training_cfg.get("batch_size", 12))
    
    num_workers = int(training_cfg.get("num_workers", 4))
    prefetch_factor = training_cfg.get("prefetch_factor", 2)
    
    pin_memory_device = ""
    if torch.cuda.is_available():
        pin_memory_device = f"cuda:{torch.cuda.current_device()}"
    
    def _worker_init_fn(worker_id: int):
        base_seed = torch.initial_seed() % 2**32
        np.random.seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(42 if not shuffle else None),
        pin_memory_device=pin_memory_device if pin_memory_device else None,
    )

