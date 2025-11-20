from __future__ import annotations
from typing import Dict, Optional
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

from models import builders as build
from models.SyntalNet import SyntalNet

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_model_with_fusion_type(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    fusion_type: Optional[str] = None,
    device: torch.device = None,
) -> SyntalNet:
    """
    Load a model, optionally swapping the multimodal fusion type.
    
    Args:
        config_path: Path to YAML config file
        checkpoint_path: Path to checkpoint file (optional)
        fusion_type: Multimodal fusion type to use (overrides config if provided)
        device: Device to load model on
    
    Returns:
        Loaded SyntalNet model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    cfg = build.config(config_path)
    
    # Override fusion type if specified
    if fusion_type is not None:
        cfg["model"]["args"]["mm_fusion_type"] = fusion_type
    
    # Build model
    model = build.model(cfg)
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if isinstance(state, dict):
            if "model" in state:
                model_state = state["model"]
            elif "state_dict" in state:
                model_state = state["state_dict"]
            else:
                model_state = state
        else:
            model_state = state
        
        # Load state dict
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        
        if len(missing) > 0:
            print(f"Warning: Missing keys in checkpoint: {missing[:10]}...")
        if len(unexpected) > 0:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected[:10]}...")
    
    model = model.to(device)
    model.eval()
    
    return model


def get_fusion_type_from_variant(variant_name: str) -> str:
    """
    Map variant name to multimodal fusion type string.
    
    Args:
        variant_name: One of "glrx", "glr_x", "uniform_avg", "gated_sum", 
                      "pairwise", "concat_mlp"
    
    Returns:
        Fusion type string for config
    """
    mapping = {
        # GLR_X variants
        "glrx": "glrx",
        "glr_x": "glrx",
        "glr": "glrx",
        
        # Baseline variants
        "uniform_avg": "uniform_avg",
        "uniform": "uniform_avg",
        "mean": "uniform_avg",
        
        "gated_sum_only": "gated_sum_only",
        "gated_sum": "gated_sum_only",
        "sum_only": "gated_sum_only",
        
        "pairwise_only": "pairwise_only",
        "pairwise": "pairwise_only",
        "pair_only": "pairwise_only",
        
        "concat_mlp": "concat_mlp",
        "concat": "concat_mlp",
        "mlp": "concat_mlp",
    }
    
    variant_lower = variant_name.lower()
    return mapping.get(variant_lower, variant_lower)


def find_checkpoint_path(
    checkpoint_dir: str,
    variant_name: str,
    split_mode: str,
    fold_idx: Optional[int] = None,
    held_out_group: Optional[int] = None,
) -> Optional[str]:
    """
    Find checkpoint path based on split mode and fold index.
    
    This function handles different checkpoint directory structures:
    - For "item", "group": looks for single checkpoint per variant
    - For "kfold", "logo": looks for per-fold checkpoints
    
    Args:
        checkpoint_dir: Base checkpoint directory
        variant_name: Model variant name (e.g., "glrx", "uniform_avg")
        split_mode: Split mode from config ("item", "group", "logo", "kfold")
        fold_idx: Fold index (required for kfold/logo mode, ignored otherwise)
        held_out_group: Held-out group ID for logo mode (if None, uses fold_idx)
    
    Returns:
        Path to checkpoint file, or None if not found
    
    Checkpoint search order:
    1. For item/group: {checkpoint_dir}/{variant_name}/best.pth
    2. For kfold: {checkpoint_dir}/{variant_name}/kfold/fold_{fold_idx:02d}/best.pth
    3. For logo: {checkpoint_dir}/{variant_name}/logo/fold_{held_out_group:02d}/best.pth
    4. Falls back to: latest.pth, epoch_100.pth, checkpoint.pth
    """
    variant_dir = os.path.join(checkpoint_dir, variant_name)
    checkpoint_names = ["best.pth", "latest.pth", "epoch_100.pth", "checkpoint.pth"]
    
    # For k-fold, checkpoints are in fold-specific subdirectories
    if split_mode == "kfold":
        if fold_idx is None:
            raise ValueError("fold_idx must be provided for kfold split mode")
        
        # Try kfold/fold_XX structure (standard from training)
        fold_tag = f"fold_{fold_idx:02d}"
        fold_dir = os.path.join(variant_dir, "kfold", fold_tag)
        
        for ckpt_name in checkpoint_names:
            ckpt_path = os.path.join(fold_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        # Alternative: try direct fold_XX in variant directory
        fold_dir_alt = os.path.join(variant_dir, fold_tag)
        for ckpt_name in checkpoint_names:
            ckpt_path = os.path.join(fold_dir_alt, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        # Alternative: try fold-specific checkpoint names in variant directory
        for ckpt_name in checkpoint_names:
            base_name = os.path.splitext(ckpt_name)[0]
            ckpt_path = os.path.join(variant_dir, f"{base_name}_fold_{fold_idx:02d}.pth")
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        return None
    
    # For logo (leave-one-group-out), checkpoints are also per-fold
    elif split_mode == "logo":
        # Try logo/fold_XX structure (standard from training)
        fold_tag = f"fold_{fold_idx:02d}"
        fold_dir = os.path.join(variant_dir, "logo", fold_tag)
        
        for ckpt_name in checkpoint_names:
            ckpt_path = os.path.join(fold_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        # If held_out_group is provided, use it
        if held_out_group is not None:
            fold_tag = f"fold_{held_out_group:02d}"
            fold_dir = os.path.join(variant_dir, "logo", fold_tag)
            for ckpt_name in checkpoint_names:
                ckpt_path = os.path.join(fold_dir, ckpt_name)
                if os.path.exists(ckpt_path):
                    return ckpt_path
        
        # Try to discover available logo folds
        logo_dir = os.path.join(variant_dir, "logo")
        if os.path.exists(logo_dir):
            # Get all fold directories, sorted
            fold_dirs = sorted([d for d in os.listdir(logo_dir) if d.startswith("fold_")])
            if fold_idx < len(fold_dirs):
                fold_dir = os.path.join(logo_dir, fold_dirs[fold_idx])
                for ckpt_name in checkpoint_names:
                    ckpt_path = os.path.join(fold_dir, ckpt_name)
                    if os.path.exists(ckpt_path):
                        return ckpt_path
        
        return None
    
    else:
        # For item, group: single checkpoint per variant
        for ckpt_name in checkpoint_names:
            ckpt_path = os.path.join(variant_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        return None

