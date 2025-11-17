from __future__ import annotations

import argparse
import csv
import os
import random
import yaml
import glob
import re
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset

from data.dataset import GroupDynamicsDataset
from data.collate import collate_fn
from data.transforms import StandardizeTransform
from engine.utils import BuildAutocastKWargs, modalities_to_branches
import models.builders as build
from torch.amp import autocast


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run inference and save predictions to CSV")
    p.add_argument("--config", required=True, type=str, help="Path to config YAML file")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint")
    p.add_argument("--out", required=True, type=str, help="Output CSV file path")
    p.add_argument("--save-probs", action="store_true", help="Also save prediction probabilities")
    return p.parse_args()


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
            
        fold_idx = int(fold_idx)
        full = GroupDynamicsDataset(**args)
        _, val_subset = _create_kfold_splits(full, n_folds, fold_idx, split_seed)
        return val_subset

    else:
        raise ValueError(f"Unknown split.mode: {mode}")


def collate_fn_with_indices(batch, dataset_indices=None, dataset_samples=None):
    """
    Custom collate function that also returns sample indices and group IDs.
    Expects batch items: (modality_data, modality_mask, labels, gid)
    
    Args:
        batch: list of tuples from dataset
        dataset_indices: list of dataset indices for this batch (if using Subset)
        dataset_samples: reference to dataset.samples (for accessing start_time)
    """
    # Extract group IDs and sample metadata before calling standard collate
    sample_info = []
    for i, sample in enumerate(batch):
        # sample is (modality_data, modality_mask, labels, gid)
        gid = sample[3] if len(sample) >= 4 else -1
        
        # Get dataset index (original if using Subset, otherwise just the index in batch)
        dataset_idx = dataset_indices[i] if dataset_indices is not None else None
        
        # Get start_time from dataset samples if available
        start_time = None
        if dataset_samples is not None and dataset_idx is not None:
            if dataset_idx < len(dataset_samples):
                start_time = dataset_samples[dataset_idx].get('start_time', None)
        
        sample_info.append({
            'group_id': gid,
            'dataset_idx': dataset_idx,
            'start_time': start_time,
        })
    
    # Create batch without gid for standard collate
    batch_for_collate = [(sample[0], sample[1], sample[2]) for sample in batch]
    batch_data, batch_labels = collate_fn(batch_for_collate)
    
    return batch_data, batch_labels, sample_info


# ----------------------------- Inference -----------------------------
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    autocast_kwargs: dict,
    save_probs: bool = False,
    val_bs: int = 8,
    dataset: Optional[GroupDynamicsDataset | Subset] = None,
) -> List[Dict[str, Any]]:
    """
    Run inference and collect predictions with ground truth.
    Returns a list of dictionaries, one per prediction.
    """
    model.eval()
    results = []
    
    # Get head names from model
    model_unwrapped = getattr(model, "module", model)
    ind_head_names = []
    grp_head_names = []
    
    if getattr(model_unwrapped, "individual_classifier", None) is not None:
        ind_head_names = list(model_unwrapped.individual_classifier.classifiers.keys())
    if getattr(model_unwrapped, "group_classifier", None) is not None:
        grp_head_names = list(model_unwrapped.group_classifier.classifiers.keys())
    
    # Get dataset reference for accessing samples
    dataset_ref = dataset
    if isinstance(dataset, Subset):
        # If using Subset, get the underlying dataset
        dataset_ref = dataset.dataset
        subset_indices = dataset.indices
    else:
        subset_indices = None
    
    # Get samples reference for start_time
    dataset_samples = None
    if hasattr(dataset_ref, 'samples'):
        dataset_samples = dataset_ref.samples
    
    with torch.no_grad():
        for batch_idx, (batch_data, batch_labels, sample_info) in enumerate(dataloader):
            batch_data = {k: (v[0].to(device), v[1].to(device)) for k, v in batch_data.items()}
            batch_data = modalities_to_branches(batch_data)
            batch_labels = {k: v.to(device) for k, v in batch_labels.items()}
            
            with autocast(**autocast_kwargs):
                _, logits = model(batch_data)
            
            # Determine batch size from labels
            if "group" in batch_labels:
                B = batch_labels["group"].shape[0]
            elif "individual" in batch_labels:
                B = batch_labels["individual"].shape[0]
            else:
                B = 0
            
            if "individual" in batch_labels:
                B_ind, P, L_ind = batch_labels["individual"].shape
            else:
                B_ind, P, L_ind = 0, 3, 0
            
            # Process individual predictions
            if "individual" in logits and "individual" in batch_labels:
                ind_labels = batch_labels["individual"]  # (B, P, L)
                ind_logits_dict = logits["individual"]  # {head_name: (B*P, K)}
                
                # Flatten for matching with logits
                ind_labels_flat = ind_labels.view(B_ind * P, L_ind).cpu().numpy()  # (B*P, L)
                
                for head_idx, head_name in enumerate(ind_head_names):
                    if head_name not in ind_logits_dict:
                        continue
                    
                    lg = ind_logits_dict[head_name].cpu()  # (B*P, K)
                    probs = torch.softmax(lg, dim=-1).numpy()
                    preds = np.argmax(probs, axis=-1)  # (B*P,)
                    
                    for flat_idx in range(B_ind * P):
                        batch_idx_local = flat_idx // P
                        person_idx = flat_idx % P
                        
                        if batch_idx_local < len(sample_info):
                            sample_idx_info = sample_info[batch_idx_local]
                            gid = sample_idx_info.get('group_id', -1)
                        else:
                            gid = -1
                        
                        gt = int(ind_labels_flat[flat_idx, head_idx]) if head_idx < L_ind else -1
                        pred = int(preds[flat_idx])
                        
                        # Get dataset info from sample_info
                        sample_meta = sample_info[batch_idx_local] if batch_idx_local < len(sample_info) else {}
                        dataset_idx = sample_meta.get('dataset_idx', None)
                        start_time = sample_meta.get('start_time', None)
                        
                        result = {
                            'sample_idx': batch_idx * val_bs + batch_idx_local,
                            'dataset_idx': dataset_idx if dataset_idx is not None else batch_idx * val_bs + batch_idx_local,
                            'batch_idx': batch_idx,
                            'person_idx': person_idx,
                            'group_id': gid,
                            'start_time': start_time,
                            'split': 'individual',
                            'head_name': head_name,
                            'ground_truth': gt,
                            'prediction': pred,
                        }
                        
                        if save_probs:
                            result['prob_class_0'] = float(probs[flat_idx, 0])
                            result['prob_class_1'] = float(probs[flat_idx, 1])
                            if probs.shape[1] > 2:
                                result['prob_class_2'] = float(probs[flat_idx, 2])
                        
                        results.append(result)
            
            # Process group predictions
            if "group" in logits and "group" in batch_labels:
                grp_labels = batch_labels["group"].cpu().numpy()  # (B, L)
                grp_logits_dict = logits["group"]  # {head_name: (B, K)}
                
                for head_idx, head_name in enumerate(grp_head_names):
                    if head_name not in grp_logits_dict:
                        continue
                    
                    lg = grp_logits_dict[head_name].cpu()  # (B, K)
                    probs = torch.softmax(lg, dim=-1).numpy()
                    preds = np.argmax(probs, axis=-1)  # (B,)
                    
                    for batch_idx_local in range(B):
                        if batch_idx_local < len(sample_info):
                            sample_idx_info = sample_info[batch_idx_local]
                            gid = sample_idx_info.get('group_id', -1)
                        else:
                            gid = -1
                        
                        gt = int(grp_labels[batch_idx_local, head_idx]) if head_idx < grp_labels.shape[1] else -1
                        pred = int(preds[batch_idx_local])
                        
                        # Get dataset info from sample_info
                        sample_meta = sample_info[batch_idx_local] if batch_idx_local < len(sample_info) else {}
                        dataset_idx = sample_meta.get('dataset_idx', None)
                        start_time = sample_meta.get('start_time', None)
                        
                        result = {
                            'sample_idx': batch_idx * val_bs + batch_idx_local,
                            'dataset_idx': dataset_idx if dataset_idx is not None else batch_idx * val_bs + batch_idx_local,
                            'batch_idx': batch_idx,
                            'person_idx': -1,  # N/A for group
                            'group_id': gid,
                            'start_time': start_time,
                            'split': 'group',
                            'head_name': head_name,
                            'ground_truth': gt,
                            'prediction': pred,
                        }
                        
                        if save_probs:
                            result['prob_class_0'] = float(probs[batch_idx_local, 0])
                            result['prob_class_1'] = float(probs[batch_idx_local, 1])
                            if probs.shape[1] > 2:
                                result['prob_class_2'] = float(probs[batch_idx_local, 2])
                        
                        results.append(result)
    
    return results


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """
    Save inference results to CSV file.
    """
    if not results:
        print("Warning: No results to save!")
        return
    
    # Get all unique keys (handling variable prob columns)
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    # Sort keys for consistent column order
    base_keys = ['sample_idx', 'dataset_idx', 'batch_idx', 'person_idx', 'group_id', 
                 'start_time', 'split', 'head_name', 'ground_truth', 'prediction']
    prob_keys = sorted([k for k in all_keys if k.startswith('prob_class_')])
    fieldnames = base_keys + prob_keys
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"[OK] Saved {len(results)} predictions to {output_path}")


# ----------------------------- Main -----------------------------
def main():
    args = parse_args()
    cfg = build.config(args.config)
    
    # Set seed for reproducibility
    seed = int(cfg.get("training", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------- Data --------------------
    print("Building Dataloader...")
    val_dataset = build_val_dataset(cfg)
    
    val_bs = int(cfg["training"].get("val_batch_size", cfg["training"].get("batch_size", 8)))
    num_workers = int(cfg["training"].get("num_workers", 4))
    prefetch_factor = cfg["training"].get("prefetch_factor", 4)
    pin_memory_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else ""
    
    # We need to track dataset indices manually since DataLoader doesn't expose them easily
    # Instead, we'll create a wrapper that tracks indices
    class IndexTrackingDataset:
        def __init__(self, dataset):
            self.dataset = dataset
            if isinstance(dataset, Subset):
                self.original_dataset = dataset.dataset
                self.subset_indices = dataset.indices
                self.is_subset = True
            else:
                self.original_dataset = dataset
                self.subset_indices = None
                self.is_subset = False
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            # Add the dataset index to the item
            if self.is_subset:
                original_idx = self.subset_indices[idx]
            else:
                original_idx = idx
            # item is (modality_data, modality_mask, labels, gid)
            # We'll modify the collate function to handle this
            return item, original_idx
    
    # Wrap dataset to track indices
    indexed_dataset = IndexTrackingDataset(val_dataset)
    
    def collate_fn_with_tracking(batch):
        """Collate function that extracts dataset indices from wrapped dataset"""
        # batch is list of ((modality_data, modality_mask, labels, gid), dataset_idx)
        items = [item[0] for item in batch]
        dataset_indices = [item[1] for item in batch]
        
        # Get dataset reference
        dataset_ref = val_dataset.dataset if isinstance(val_dataset, Subset) else val_dataset
        dataset_samples = dataset_ref.samples if hasattr(dataset_ref, 'samples') else None
        
        return collate_fn_with_indices(items, dataset_indices=dataset_indices, dataset_samples=dataset_samples)
    
    loader = DataLoader(
        indexed_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn_with_tracking,
        generator=torch.Generator().manual_seed(seed),
        pin_memory_device=pin_memory_device,
    )
    
    # -------------------- Model --------------------
    print("Building Model...")
    model = build.model(cfg).to(device)
    
    state = torch.load(args.checkpoint, map_location="cpu")
    print("Loading Model Checkpoint...")
    
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"[WARNING] Missing keys: {missing}")
    if unexpected:
        print(f"[WARNING] Unexpected keys: {unexpected}")
    
    model.eval()
    
    # -------------------- Inference --------------------
    print("Running Inference...")
    autocast_kwargs = BuildAutocastKWargs(cfg, device)
    
    results = run_inference(
        model=model,
        dataloader=loader,
        device=device,
        autocast_kwargs=autocast_kwargs,
        save_probs=args.save_probs,
        val_bs=val_bs,
        dataset=val_dataset,
    )
    
    # -------------------- Save Results --------------------
    save_results_to_csv(results, args.out)
    print(f"[OK] Inference complete. Total predictions: {len(results)}")


if __name__ == "__main__":
    main()

