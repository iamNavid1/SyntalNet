from __future__ import annotations
from typing import Dict, Optional
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models import builders as build
from models.SyntalNet import SyntalNet


def load_model_with_fusion_type(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    fusion_type: Optional[str] = None,
    device: torch.device = None,
) -> SyntalNet:
    """
    Load a model, optionally swapping the fusion type.
    
    Args:
        config_path: Path to YAML config file
        checkpoint_path: Path to checkpoint file (optional)
        fusion_type: Fusion type to use (overrides config if provided)
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
        cfg["model"]["args"]["mc_fusion_type"] = fusion_type
    
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
    Map variant name to fusion type string.
    
    Args:
        variant_name: One of "bscx", "proj_only", "concat_proj", "uniform_avg"
    
    Returns:
        Fusion type string for config
    """
    mapping = {
        "bscx": "bscx",
        "bscx_full": "bscx",
        "proj_only": "bscx_proj_only",
        "bscx_proj_only": "bscx_proj_only",
        "concat_proj": "concat_proj",
        "bscx_concat_proj": "concat_proj",
        "uniform_avg": "uniform_avg",
        "bscx_uniform_avg": "uniform_avg",
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
    - For "item", "group": looks for single checkpoint
    - For "kfold", "logo": looks for per-fold checkpoints
    
    Args:
        checkpoint_dir: Base checkpoint directory (checkpoints are directly under this)
        variant_name: Model variant name (kept for compatibility, not used in path)
        split_mode: Split mode from config ("item", "group", "logo", "kfold")
        fold_idx: Fold index (required for kfold/logo mode, ignored otherwise)
        held_out_group: Held-out group ID for logo mode (if None, uses fold_idx)
    
    Returns:
        Path to checkpoint file, or None if not found
    """
    checkpoint_names = ["best.pth", "latest.pth", "epoch_100.pth", "checkpoint.pth"]
    
    # For k-fold, checkpoints are in fold-specific subdirectories
    if split_mode == "kfold":
        if fold_idx is None:
            raise ValueError("fold_idx must be provided for kfold split mode")
        
        # Try kfold/fold_XX structure (standard from training)
        fold_tag = f"fold_{fold_idx:02d}"
        fold_dir = os.path.join(checkpoint_dir, "kfold", fold_tag)
        
        for ckpt_name in checkpoint_names:
            ckpt_path = os.path.join(fold_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        # Alternative: try direct fold_XX in checkpoint directory
        fold_dir_alt = os.path.join(checkpoint_dir, fold_tag)
        for ckpt_name in checkpoint_names:
            ckpt_path = os.path.join(fold_dir_alt, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        # Alternative: try fold-specific checkpoint names in checkpoint directory
        for ckpt_name in checkpoint_names:
            base_name = os.path.splitext(ckpt_name)[0]
            ckpt_path = os.path.join(checkpoint_dir, f"{base_name}_fold_{fold_idx:02d}.pth")
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        return None
    
    # For logo (leave-one-group-out), checkpoints are also per-fold
    elif split_mode == "logo":        
        fold_tag = f"fold_{fold_idx:02d}"
        fold_dir = os.path.join(checkpoint_dir, "logo", fold_tag)
        
        for ckpt_name in checkpoint_names:
            ckpt_path = os.path.join(fold_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        if held_out_group is not None:
            fold_tag = f"fold_{held_out_group:02d}"
            fold_dir = os.path.join(checkpoint_dir, "logo", fold_tag)
            for ckpt_name in checkpoint_names:
                ckpt_path = os.path.join(fold_dir, ckpt_name)
                if os.path.exists(ckpt_path):
                    return ckpt_path
        
        logo_dir = os.path.join(checkpoint_dir, "logo")
        if os.path.exists(logo_dir):
            fold_dirs = sorted([d for d in os.listdir(logo_dir) if d.startswith("fold_")])
            if fold_idx < len(fold_dirs):
                fold_dir = os.path.join(logo_dir, fold_dirs[fold_idx])
                for ckpt_name in checkpoint_names:
                    ckpt_path = os.path.join(fold_dir, ckpt_name)
                    if os.path.exists(ckpt_path):
                        return ckpt_path
        
        return None
    
    else:
        # For item, group: single checkpoint in checkpoint directory
        for ckpt_name in checkpoint_names:
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        
        return None

