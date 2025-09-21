import argparse
import os
import sys
import yaml
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import GroupDynamicsDataset
from data.collate import collate_fn
from data.transforms import StandardizeTransform


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: Dict[str, Any], override_root: Optional[str]) -> GroupDynamicsDataset:
    ds_args = dict(cfg["dataset"]["args"]) if "dataset" in cfg and "args" in cfg["dataset"] else {}
    if override_root:
        ds_args["root_dir"] = override_root
    # force face-only modality
    ds_args["modalities"] = ["face"]

    # optional normalization stats like in train.py
    stats = cfg["dataset"].get("stats_dir") if "dataset" in cfg else None
    if isinstance(stats, str) and os.path.isfile(stats):
        with open(stats, "r") as f:
            stats = yaml.safe_load(f)
    if stats:
        ds_args["transforms"] = StandardizeTransform(stats)

    # ensure item mode semantics by not splitting; we construct full dataset directly
    return GroupDynamicsDataset(**ds_args)


def build_loader(dataset: GroupDynamicsDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )


def build_weighted_loader(dataset: GroupDynamicsDataset, batch_size: int) -> DataLoader:
    """Build a DataLoader with WeightedRandomSampler to balance class distribution."""
    # Calculate weights based on individual labels (combining all people and heads)
    weights = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        labels = sample[2]  # (modality_data, modality_mask, labels, gid)
        
        # Get individual labels for all people and heads
        individual_labels = labels['individual']  # (3, 2) - 3 people, 2 heads
        
        # Calculate weight based on class frequency (inverse frequency weighting)
        # Combine all individual labels across people and heads
        all_labels = individual_labels.view(-1)  # Flatten to (6,)
        valid_labels = all_labels[all_labels >= 0]  # Remove -1 (missing)
        
        if len(valid_labels) > 0:
            # Count occurrences of each class
            class_counts = torch.bincount(valid_labels, minlength=3)
            # Calculate inverse frequency weights
            total_valid = class_counts.sum()
            if total_valid > 0:
                # Weight = 1 / (class_count + 1) to avoid division by zero
                sample_weight = 1.0 / (class_counts.float() + 1.0)
                # Average weight across all valid labels for this sample
                avg_weight = sample_weight[valid_labels].mean().item()
            else:
                avg_weight = 1.0
        else:
            avg_weight = 1.0
        
        weights.append(avg_weight)
    
    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )


def count_classes(t: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    # t is an integer tensor possibly containing -1 for missing; count 0..num_classes-1
    valid = t.view(-1)
    valid = valid[valid >= 0]
    if valid.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.long)
    counts = torch.bincount(valid, minlength=num_classes)
    return counts[:num_classes]


def plot_class_distribution(group_counts, individual_counts, batch_size, suffix=""):
    """
    Plot class distribution per batch with separate figures for each head.
    group_counts: (num_batches, 3 heads, 3 classes) - 3 group heads
    individual_counts: (num_batches, 2 heads, 3 classes) - 2 individual heads (combined across 3 people)
    suffix: added to filename to distinguish between original and weighted sampling
    """
    num_batches = len(group_counts)
    
    # Create 5 separate figures (3 group + 2 individual heads)
    head_names = ['Q1 (Group)', 'Q2 (Group)', 'Q3 (Group)', 'Q4 (Individual)', 'Q5 (Individual)']
    
    # Plot each head separately
    for head_idx in range(5):
        plt.figure(figsize=(20, 6))  # Much wider for better batch visibility
        
        if head_idx < 3:  # Group heads
            data = group_counts[:, head_idx, :]  # (num_batches, 3 classes)
            title = f'{head_names[head_idx]} - Class Distribution per Batch (Batch Size: {batch_size})'
        else:  # Individual heads
            data = individual_counts[:, head_idx - 3, :]  # (num_batches, 3 classes)
            title = f'{head_names[head_idx]} - Class Distribution per Batch (Batch Size: {batch_size})'
        
        x = np.arange(num_batches)
        width = 0.8  # Wider bars for better visibility
        
        # Plot 3 classes for this head
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        for class_idx in range(3):
            plt.bar(x + class_idx * width, data[:, class_idx], width, 
                    label=f'Class {class_idx}', color=colors[class_idx], alpha=0.8)
        
        plt.xlabel('Batch Index')
        plt.ylabel('Number of Samples')
        plt.title(title)
        plt.xticks(x + width, [f'{i+1}' for i in range(num_batches)], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save each head separately
        base_filename = f'head_{head_idx+1}_{head_names[head_idx].replace(" ", "_").replace("(", "").replace(")", "")}_distribution'
        filename = f'{base_filename}{suffix}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"Saved 5 separate plots{suffix}:")
    for i, name in enumerate(head_names):
        base_filename = f'head_{i+1}_{name.replace(" ", "_").replace("(", "").replace(")", "")}_distribution'
        filename = f'{base_filename}{suffix}.png'
        print(f"  {filename}")


def analyze_dataset(loader, batch_size, description=""):
    """Analyze dataset and return counts for plotting."""
    print(f"\n{description}")
    print(f"Total samples: {len(loader.dataset)} | Batches: {len(loader)} | Batch size: {batch_size}")
    
    # Track each head separately
    # Group labels: 3 heads (Q1, Q2, Q3)
    total_group_heads = torch.zeros(3, 3, dtype=torch.long)  # (3 heads, 3 classes)
    # Individual labels: 2 heads (Q4, Q5) combined across 3 people
    total_individual_heads = torch.zeros(2, 3, dtype=torch.long)  # (2 heads, 3 classes)
    
    # Store counts for plotting
    group_counts_per_batch = []
    individual_counts_per_batch = []

    for batch_idx, (_, batch_labels) in enumerate(loader):
        # group labels: shape (B, 3) - 3 group heads
        grp = batch_labels["group"].to(torch.long)  # (B, 3)
        # individual labels: shape (B, 3, 2) - 3 people, 2 individual heads
        ind = batch_labels["individual"].to(torch.long)  # (B, 3, 2)
        
        # Count for each group head separately
        grp_counts_per_head = []
        for head_idx in range(3):
            head_data = grp[:, head_idx]  # (B,)
            counts = count_classes(head_data, 3)
            grp_counts_per_head.append(counts)
            total_group_heads[head_idx] += counts
        
        # Count for each individual head separately (combine all 3 people)
        ind_counts_per_head = []
        for head_idx in range(2):
            head_data = ind[:, :, head_idx]  # (B, 3) - all people for this head
            counts = count_classes(head_data, 3)
            ind_counts_per_head.append(counts)
            total_individual_heads[head_idx] += counts
        
        # Convert to numpy for plotting
        grp_counts_array = torch.stack(grp_counts_per_head, dim=0).numpy()  # (3, 3)
        ind_counts_array = torch.stack(ind_counts_per_head, dim=0).numpy()  # (2, 3)
        
        group_counts_per_batch.append(grp_counts_array)
        individual_counts_per_batch.append(ind_counts_array)
        
        print(f"\rProcessing batch {batch_idx+1}/{len(loader)}...", end="")
        if batch_idx == len(loader) - 1:
            print()

    # Convert to numpy arrays for plotting
    group_counts_per_batch = np.array(group_counts_per_batch)  # (num_batches, 3, 3)
    individual_counts_per_batch = np.array(individual_counts_per_batch)  # (num_batches, 2, 3)
    
    print("=" * 50)
    print("OVERALL TOTALS:")
    print("Group Labels (3 heads):")
    for i in range(3):
        print(f"  Head {i+1} (Q{i+1}): {total_group_heads[i].tolist()}")
    
    print("Individual Labels (2 heads, combined across 3 people):")
    for i in range(2):
        print(f"  Head {i+1} (Q{i+4}): {total_individual_heads[i].tolist()}")
    
    return group_counts_per_batch, individual_counts_per_batch


def main():
    parser = argparse.ArgumentParser(description="Inspect class balance per batch for face modality")
    parser.add_argument("--config", type=str, default="./configs/kernel.yaml", help="Path to YAML config")
    parser.add_argument("--root", type=str, default=None, help="Override dataset root_dir")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_dataset(cfg, args.root)
    
    print("=" * 80)
    print("ANALYSIS 1: ORIGINAL DATASET (Sequential Order)")
    print("=" * 80)
    
    # Analyze original dataset
    loader = build_loader(dataset, args.batch_size)
    group_counts_orig, individual_counts_orig = analyze_dataset(loader, args.batch_size, "Original Dataset Analysis")
    
    # Create visualization for original dataset
    print("\nGenerating visualization for original dataset...")
    plot_class_distribution(group_counts_orig, individual_counts_orig, args.batch_size, "_original")
    
    print("\n" + "=" * 80)
    print("ANALYSIS 2: WEIGHTED RANDOM SAMPLING (Balanced Classes)")
    print("=" * 80)
    
    # Analyze weighted dataset
    weighted_loader = build_weighted_loader(dataset, args.batch_size)
    group_counts_weighted, individual_counts_weighted = analyze_dataset(weighted_loader, args.batch_size, "Weighted Sampling Analysis")
    
    # Create visualization for weighted dataset
    print("\nGenerating visualization for weighted sampling...")
    plot_class_distribution(group_counts_weighted, individual_counts_weighted, args.batch_size, "_weighted")
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print("Original vs Weighted Sampling - Class Distribution Comparison:")
    print("(This shows how WeightedRandomSampler affects class balance)")
    
    # Compare the two approaches
    print("\nGroup Labels Comparison:")
    for i in range(3):
        orig_totals = group_counts_orig.sum(axis=0)[i, :]
        weighted_totals = group_counts_weighted.sum(axis=0)[i, :]
        print(f"  Q{i+1}: Original {orig_totals.tolist()} vs Weighted {weighted_totals.tolist()}")
    
    print("\nIndividual Labels Comparison:")
    for i in range(2):
        orig_totals = individual_counts_orig.sum(axis=0)[i, :]
        weighted_totals = individual_counts_weighted.sum(axis=0)[i, :]
        print(f"  Q{i+4}: Original {orig_totals.tolist()} vs Weighted {weighted_totals.tolist()}")


if __name__ == "__main__":
    main()


