"""
Model Loader for Reliability-Switch Experiments

Handles loading trained SyntalNet checkpoints, freezing encoders and branch mixers,
and rebuilding the multimodal fusion module with a specific variant.
"""

from __future__ import annotations

import copy
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

import models.builders as build
from models.SyntalNet import SyntalNet


def load_base_config(config_path: str) -> Dict:
    """Load the baseline config that was used to train SyntalNet."""
    cfg = build.config(config_path)
    if "model" not in cfg:
        raise ValueError(f"Config at {config_path} is missing the 'model' section.")
    return cfg


def build_variant_model(base_cfg: Dict, fusion_type: str) -> SyntalNet:
    """
    Instantiate a SyntalNet variant with a specific multimodal fusion module.
    The incoming config dict is deep-copied to avoid accidental mutation.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("model", {}).setdefault("args", {})
    cfg["model"]["args"]["mm_fusion_type"] = fusion_type
    model = build.model(cfg)
    if not isinstance(model, SyntalNet):
        raise TypeError("Expected build.model to return SyntalNet.")
    return model


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle checkpoints saved from DDP models."""
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_backbone_weights(model: SyntalNet, checkpoint_path: str) -> Tuple[List[str], List[str]]:
    """Load a pretrained checkpoint (frozen backbone) into the model."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def freeze_backbone_except_fusion_and_heads(model: SyntalNet) -> List[str]:
    """
    Freeze all parameters except multimodal fusion + classification heads.
    Returns the list of parameter names that remain trainable.
    """
    for p in model.parameters():
        p.requires_grad = False

    trainable: List[str] = []

    if getattr(model, "mm_fusion", None) is not None:
        for name, param in model.mm_fusion.named_parameters():
            param.requires_grad = True
            trainable.append(f"mm_fusion.{name}")

    for head_name in ("individual_classifier", "group_classifier"):
        classifier = getattr(model, head_name, None)
        if classifier is None:
            continue
        for name, param in classifier.named_parameters():
            param.requires_grad = True
            trainable.append(f"{head_name}.{name}")

    return trainable


def ensure_dirs(*paths: str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_checkpoint_for_reliability_training(
    checkpoint_path: str,
    config: Dict[str, Any],
    fusion_variant: str,
    fusion_kwargs: Optional[Dict[str, Any]] = None,
    device: torch.device = torch.device('cpu'),
) -> Tuple[SyntalNet, int]:
    """
    Load a trained SyntalNet checkpoint and prepare it for reliability-switch training.
    
    Args:
        checkpoint_path: Path to SyntalNet checkpoint (.pth file)
        config: Original SyntalNet config
        fusion_variant: Name of fusion variant to use ("glrx", "uniform_avg", "concat_mlp")
        fusion_kwargs: Additional kwargs for fusion module (currently unused)
        device: Device to load model on
        
    Returns:
        Tuple of (model, start_epoch)
          - model: SyntalNet with new fusion module and frozen backbone
          - start_epoch: Epoch from which to resume (always 0 for new fusion training)
    """
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Build model with the desired fusion variant
    model = build_variant_model(config, fusion_variant)
    model = model.to(device)
    
    # Load backbone weights (branches + heads) from checkpoint
    # The mm_fusion will have mismatched keys (expected for new fusion variant)
    missing, unexpected = load_backbone_weights(model, checkpoint_path)
    
    # Filter out expected mm_fusion mismatches from the warnings
    mm_fusion_missing = [k for k in missing if k.startswith("mm_fusion.")]
    other_missing = [k for k in missing if not k.startswith("mm_fusion.")]
    mm_fusion_unexpected = [k for k in unexpected if k.startswith("mm_fusion.")]
    other_unexpected = [k for k in unexpected if not k.startswith("mm_fusion.")]
    
    if other_missing:
        print(f"Warning: Missing keys (non-fusion): {len(other_missing)}")
        if len(other_missing) <= 10:
            for k in other_missing:
                print(f"  - {k}")
    
    if other_unexpected:
        print(f"Warning: Unexpected keys (non-fusion): {len(other_unexpected)}")
        if len(other_unexpected) <= 10:
            for k in other_unexpected:
                print(f"  - {k}")
    
    print(f"Loaded backbone from checkpoint (mm_fusion initialized randomly)")
    print(f"  mm_fusion params: {len(mm_fusion_missing)} new, {len(mm_fusion_unexpected)} old")
    
    # Freeze everything except mm_fusion and classification heads
    trainable = freeze_backbone_except_fusion_and_heads(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Model prepared for reliability-switch training:")
    print(f"  Fusion variant: {fusion_variant}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"  Trainable param groups: mm_fusion, individual_classifier, group_classifier")
    
    return model, 0  # Always start from epoch 0 for new fusion training


def find_last_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the last checkpoint in a directory (highest epoch number)."""
    import re
    import glob
    
    pattern = os.path.join(checkpoint_dir, "epoch_*.pth")
    ckpts = sorted(glob.glob(pattern))
    
    def _epoch_num(p):
        m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    
    ckpts.sort(key=_epoch_num)
    return ckpts[-1] if ckpts else None

