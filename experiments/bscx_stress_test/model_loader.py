"""
Model loader for stress testing different fusion variants.

This module loads checkpoints and builds models with different fusion types.
"""

from __future__ import annotations
from typing import Dict, Optional
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
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

