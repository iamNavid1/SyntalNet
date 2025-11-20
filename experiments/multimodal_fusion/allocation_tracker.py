from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from engine.utils import modalities_to_branches, BuildAutocastKWargs
from experiments.multimodal_fusion.corruptions import modality_noise

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class AllocationTracker:
    """
    Tracks allocation behavior under varying modality noise levels.
    
    This tracker runs a dedicated experiment where each modality is
    systematically corrupted with noise while recording the allocation
    weights assigned by the fusion module.
    """
    def __init__(
        self,
        device: torch.device,
        cfg: dict,
        noise_levels: Optional[List[float]] = None,
    ):
        self.device = device
        self.cfg = cfg
        self.noise_levels = noise_levels or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.autocast_kwargs = BuildAutocastKWargs(cfg, device)
    
    @torch.no_grad()
    def track_allocations(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Run allocation tracking experiment on a single fold.
        
        For each modality m and noise level n:
        1. Add noise to modality m only
        2. Run forward pass through model
        3. Extract allocation weights from GLR_X
        4. Record allocation patterns
        
        Args:
            model: SyntalNet model with GLR_X fusion
            dataloader: Validation dataloader
        
        Returns:
            Dict containing allocation patterns for each modality/noise combination
        """
        model.eval()
        model = model.to(self.device)
        
        # Determine number of modalities (branches)
        num_modalities = len(model.branches)
        
        # Check if model has allocation extraction capability
        has_allocation = (
            hasattr(model, "mm_fusion") and 
            model.mm_fusion is not None and
            hasattr(model.mm_fusion, "get_aux")
        )
        
        if not has_allocation:
            print("Warning: Model does not support allocation extraction")
            return {}
        
        # Initialize results storage
        # Structure: {modality_idx: {noise_level: [allocation_weights_per_batch]}}
        results = {
            mod_idx: {noise: [] for noise in self.noise_levels}
            for mod_idx in range(num_modalities)
        }
        
        # For each modality to corrupt
        for noisy_mod_idx in range(num_modalities):
            # For each noise level
            for noise_level in self.noise_levels:
                # Collect allocations for this (modality, noise_level) pair
                allocations_batch = []
                
                for batch_data, batch_labels in dataloader:
                    batch_data = {
                        k: (v[0].to(self.device), v[1].to(self.device))
                        for k, v in batch_data.items()
                    }
                    batch_data = modalities_to_branches(batch_data)
                    
                    # Extract branch outputs
                    z_branch = []
                    for bname, branch in model.branches.items():
                        x_dict = {m: batch_data[bname.lower()][m][0] for m in branch.mods}
                        m_dict = {m: batch_data[bname.lower()][m][1] for m in branch.mods}
                        z_b = branch(x_dict, m_dict)
                        z_branch.append(z_b)
                    
                    # Apply noise to target modality only
                    if noise_level > 0:
                        z_noisy = [z.clone() for z in z_branch]
                        z_target = z_noisy[noisy_mod_idx]
                        
                        # Add noise
                        B, D = z_target.shape
                        mean = z_target.mean(dim=1, keepdim=True)
                        var = ((z_target - mean) ** 2).mean(dim=1, keepdim=True)
                        std = var.clamp_min(1e-6).sqrt()
                        sigma = noise_level * std
                        eps = torch.randn_like(z_target)
                        z_noisy[noisy_mod_idx] = z_target + eps * sigma
                        z_branch = z_noisy
                    
                    # Run through fusion
                    with autocast(**self.autocast_kwargs):
                        _ = model.mm_fusion(z_branch)
                    
                    # Extract allocation info
                    aux = model.mm_fusion.get_aux()
                    if aux is not None and "alloc" in aux:
                        # aux["alloc"]: (B, M) on CPU
                        alloc = aux["alloc"]  # (B, M)
                        allocations_batch.append(alloc.numpy())
                
                # Aggregate allocations across batches
                if allocations_batch:
                    # Concatenate all batches: (total_samples, M)
                    alloc_all = np.concatenate(allocations_batch, axis=0)
                    # Compute mean allocation per modality
                    mean_alloc = alloc_all.mean(axis=0)  # (M,)
                    results[noisy_mod_idx][noise_level] = {
                        "mean": mean_alloc.tolist(),
                        "std": alloc_all.std(axis=0).tolist(),
                        "n_samples": len(alloc_all),
                    }
        
        return results
    
    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        variant_name: str,
        fold_idx: int,
    ):
        """
        Export allocation tracking results to CSV.
        
        Args:
            results: Results from track_allocations()
            output_path: Path to output CSV file
            variant_name: Name of model variant
            fold_idx: Fold index
        """
        rows = []
        
        for noisy_mod_idx, noise_dict in results.items():
            for noise_level, alloc_data in noise_dict.items():
                if not alloc_data:
                    continue
                
                mean_alloc = alloc_data["mean"]
                std_alloc = alloc_data["std"]
                n_samples = alloc_data["n_samples"]
                
                # Create one row per modality allocation
                for mod_idx, (mean_val, std_val) in enumerate(zip(mean_alloc, std_alloc)):
                    rows.append({
                        "variant": variant_name,
                        "fold": fold_idx,
                        "noisy_modality": noisy_mod_idx,
                        "noise_level": noise_level,
                        "target_modality": mod_idx,
                        "allocation_mean": mean_val,
                        "allocation_std": std_val,
                        "n_samples": n_samples,
                    })
        
        # Write to CSV
        if rows:
            fieldnames = [
                "variant", "fold", "noisy_modality", "noise_level",
                "target_modality", "allocation_mean", "allocation_std", "n_samples"
            ]
            
            # Create directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


def run_allocation_tracking(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cfg: dict,
    variant_name: str,
    fold_idx: int,
    output_dir: str,
    noise_levels: Optional[List[float]] = None,
) -> str:
    """
    Convenience function to run allocation tracking and export results.
    
    Args:
        model: SyntalNet model with GLR_X fusion
        dataloader: Validation dataloader
        device: Device to run on
        cfg: Config dict
        variant_name: Name of model variant
        fold_idx: Fold index
        output_dir: Output directory for CSV
        noise_levels: List of noise levels to test
    
    Returns:
        Path to exported CSV file
    """
    tracker = AllocationTracker(device, cfg, noise_levels)
    results = tracker.track_allocations(model, dataloader)
    
    output_path = os.path.join(
        output_dir,
        f"allocation_tracking_{variant_name}_fold_{fold_idx:02d}.csv"
    )
    
    tracker.export_results(results, output_path, variant_name, fold_idx)
    
    return output_path

