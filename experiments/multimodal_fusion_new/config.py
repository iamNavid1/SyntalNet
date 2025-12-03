"""
Configuration management for reliability-switch experiments.
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path


DEFAULT_CONFIG = {
    # Base SyntalNet checkpoint info
    "base_checkpoint_dir": "checkpoints/SyntalNet",
    "base_config_path": "configs/SyntalNet.yaml",
    
    # Fusion variant
    "fusion_variant": "glrx",  # One of: glrx, uniform_avg, concat_mlp
    "fusion_kwargs": {
        "rank_pair": 12,
        "alloc_hidden": 24,
        "eps_floor": 0.03,
        "p_drop": 0.1,
    },
    
    # Reliability-switch training
    "reliability_training": {
        "num_epochs": 30,
        "p_corrupt": 0.7,
        "corruption_strengths": {
            "dropout": [0.1, 0.5],
            "noise": [0.1, 0.5],
            "shuffle": [0.1, 0.5],
        },
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "scheduler": "reduce_on_plateau",
        "scheduler_patience": 3,
        "scheduler_factor": 0.5,
        "min_lr": 1e-6,
        "validate_interval": 1,
        "save_interval": 5,
    },
    
    # Stress testing
    "stress_test": {
        "corruption_configs": {
            "dropout": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
            "noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
            "shuffle": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
        },
        "per_modality": True,
    },
    
    # Logging
    "logging": {
        "log_dir": "logs/multimodal_fusion_reliability",
        "checkpoint_dir": "checkpoints/multimodal_fusion_reliability",
        "results_dir": "experiments/multimodal_fusion_reliability_results",
    },
    
    # Cross-validation
    "cv_mode": "kfold",  # One of: kfold, logo, single
    "n_folds": 5,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or return defaults.
    
    Args:
        config_path: Path to YAML config file. If None, returns defaults.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        return DEFAULT_CONFIG.copy()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge with defaults (config overrides defaults)
    merged = DEFAULT_CONFIG.copy()
    merged.update(config)
    
    return merged


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_fold_dirs(
    base_log_dir: str,
    base_checkpoint_dir: str,
    base_results_dir: str,
    fusion_variant: str,
    fold_idx: int,
    cv_mode: str = "kfold",
) -> tuple:
    """
    Get directories for a specific fold.
    
    Returns:
        (log_dir, checkpoint_dir, results_dir)
    """
    fold_tag = f"fold_{fold_idx:02d}"
    variant_tag = fusion_variant.lower()
    
    log_dir = Path(base_log_dir) / cv_mode / variant_tag / fold_tag
    checkpoint_dir = Path(base_checkpoint_dir) / cv_mode / variant_tag / fold_tag
    results_dir = Path(base_results_dir) / cv_mode / variant_tag / fold_tag
    
    return str(log_dir), str(checkpoint_dir), str(results_dir)


def get_base_checkpoint_path(
    base_checkpoint_dir: str,
    fold_idx: int,
    cv_mode: str = "kfold",
) -> str:
    """
    Get path to base SyntalNet checkpoint for a specific fold.
    
    Args:
        base_checkpoint_dir: Base checkpoint directory of trained SyntalNet
        fold_idx: Fold index
        cv_mode: Cross-validation mode (kfold or logo)
        
    Returns:
        Path to last epoch checkpoint (or best_model.pth if available)
    """
    fold_tag = f"fold_{fold_idx:02d}"
    fold_dir = Path(base_checkpoint_dir) / cv_mode / fold_tag
    
    # Try to find best_model.pth first
    best_path = fold_dir / "best_model.pth"
    if best_path.exists():
        return str(best_path)
    
    # Otherwise, find last epoch
    import glob
    import re
    
    pattern = str(fold_dir / "epoch_*.pth")
    ckpts = sorted(glob.glob(pattern))
    
    def _epoch_num(p):
        m = re.search(r"epoch_(\d+)\.pth$", Path(p).name)
        return int(m.group(1)) if m else -1
    
    ckpts.sort(key=_epoch_num)
    
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {fold_dir}")
    
    return ckpts[-1]

