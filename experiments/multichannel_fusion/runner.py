from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import sys
import time
import logging
import gc
import yaml
import random
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import models.builders as build
from experiments.multichannel_fusion.model_loader import (
    load_model_with_fusion_type,
    get_fusion_type_from_variant,
    find_checkpoint_path,
)
from experiments.common.dataset_builder import build_datasets, build_dataloader
from experiments.multichannel_fusion.evaluator import StressTestEvaluator
from experiments.common.metrics_collector import MetricsCollector


# Corruption sweep configurations
CORRUPTION_SWEEPS = {
    "stream_dropout": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
    "channel_dropout": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
    "temporal_band": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
    "jitter": [0, 5, 10, 15, 20, 25],
    "noise": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
}

# Model variants to evaluate
MODEL_VARIANTS = ["bscx", "concat_proj", "uniform_avg"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "stress_test.log")
    
    # Create logger
    logger = logging.getLogger("stress_test")
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(fh_formatter)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def is_baseline(corruption_type: str, param: float | int) -> bool:
    """Check if a corruption configuration is baseline (no corruption)."""
    baseline_conditions = {
        "stream_dropout": param == 0.0,
        "channel_dropout": param == 0.0,
        "temporal_band": param == 0.0,
        "jitter": param == 0,
        "energy": param == 1.0,
        "noise": param == 0.0,
    }
    return baseline_conditions.get(corruption_type, False)


def get_variant_name_from_config(config_path: str) -> str:
    """
    Extract variant name from config file.
    First tries to extract from filename (e.g., 'EXPT_BSCX_concat-proj.yaml' -> 'concat-proj'),
    then falls back to mc_fusion_type from config content.
    """
    basename = os.path.basename(config_path)
    name = os.path.splitext(basename)[0]
    
    # Try to extract from filename pattern EXPT_BSCX_* or EXPT_*
    if "EXPT_BSCX_" in name:
        variant = name.replace("EXPT_BSCX_", "")
        variant = variant.replace("-", "_")
        return variant
    
    # Fallback: try to get from config content
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        mc_fusion_type = cfg.get("model", {}).get("args", {}).get("mc_fusion_type", None)
        if mc_fusion_type:
            variant_mapping = {
                "bscx": "bscx",
                "bscx_proj_only": "proj_only",
                "concat_proj": "concat_proj",
                "uniform_avg": "uniform_avg",
            }
            return variant_mapping.get(mc_fusion_type, mc_fusion_type)
    except Exception:
        pass
    
    # Final fallback: use filename without extension
    return name


def get_checkpoint_dir_from_config(config_path: str) -> str:
    """
    Extract checkpoint directory from config file.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    checkpoint_dir = cfg.get("logging", {}).get("checkpoint_dir", None)
    if checkpoint_dir is None:
        raise ValueError(f"No checkpoint_dir found in config: {config_path}")
    
    return checkpoint_dir


def run_stress_test(
    config_paths: List[str],
    output_dir: str,
    variants: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
):
    """
    Run complete stress test evaluation.
    
    Args:
        config_paths: List of paths to config YAML files (one per variant)
        output_dir: Directory to save results
        variants: List of variant names to evaluate (default: all configs)
                  If provided, filters configs to only those matching the variants
        device: Device to run on (default: auto-detect)
    
    Note:
        The split mode and number of folds are determined by the config file's
        dataset.split configuration, ensuring consistency with training/evaluation splits.
        Supported modes: "item", "group", "logo", "kfold"
        Checkpoint directories are automatically extracted from each config file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("=" * 80)
    logger.info("Starting Stress Test Evaluation")
    logger.info("=" * 80)
    logger.info(f"Config files: {config_paths}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info("")
    
    # Process configs: extract variant names and checkpoint directories
    config_info: List[Tuple[str, str, str]] = []  # (config_path, variant_name, checkpoint_dir)
    
    for config_path in config_paths:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, skipping...")
            continue
        
        try:
            variant_name = get_variant_name_from_config(config_path)
            checkpoint_dir = get_checkpoint_dir_from_config(config_path)
            config_info.append((config_path, variant_name, checkpoint_dir))
            logger.info(f"  {config_path} -> variant: {variant_name}, checkpoint_dir: {checkpoint_dir}")
        except Exception as e:
            logger.error(f"Failed to process config {config_path}: {e}")
            logger.warning(f"Skipping config: {config_path}")
            continue
    
    if not config_info:
        logger.error("No valid config files found!")
        return
    
    # Filter by variants if provided
    if variants is not None:
        # Normalize variant names for comparison
        variants_normalized = [v.lower().replace("-", "_") for v in variants]
        filtered_config_info = []
        for config_path, variant_name, checkpoint_dir in config_info:
            variant_normalized = variant_name.lower().replace("-", "_")
            if variant_normalized in variants_normalized:
                filtered_config_info.append((config_path, variant_name, checkpoint_dir))
            else:
                logger.info(f"  Skipping {variant_name} (not in requested variants)")
        
        if not filtered_config_info:
            logger.error("No configs match the requested variants!")
            return
        
        config_info = filtered_config_info
        logger.info(f"Filtered to {len(config_info)} config(s) matching requested variants")
    
    logger.info(f"Evaluating {len(config_info)} variant(s)")
    logger.info("")
    
    # Use first config for dataset building (assuming all configs have same dataset settings)
    first_config_path, _, _ = config_info[0]
    cfg = build.config(first_config_path)

    training_seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(training_seed)
    
    # Build datasets (once for all variants) based on config split mode
    split_cfg = cfg.get("dataset", {}).get("split", {})
    split_mode = split_cfg.get("mode", "kfold")
    logger.info(f"Building datasets (split mode: {split_mode})...")
    folds = build_datasets(cfg)
    n_folds = len(folds)
    logger.info(f"Created {n_folds} fold(s)")
    logger.info("")
    
    # Initialize metrics collector
    collector = MetricsCollector()
    
    # Track baseline results per variant per fold (reused across corruption types)
    baseline_cache: Dict[Tuple[str, int], Dict] = {}
    
    # Evaluate each variant
    total_runs = 0
    start_time = time.time()
    
    for variant_idx, (config_path, variant_name, checkpoint_dir) in enumerate(config_info):
        logger.info("=" * 80)
        logger.info(f"Evaluating variant: {variant_name} ({variant_idx + 1}/{len(config_info)})")
        logger.info("=" * 80)
        logger.info(f"Config: {config_path}")
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        
        # Get fusion type for this variant
        fusion_type = get_fusion_type_from_variant(variant_name)
        
        # Load config for this variant (for model loading)
        variant_cfg = build.config(config_path)
        
        # For k-fold and logo, we need to load a model per fold
        # For item/group splits, we load one model for all folds
        if split_mode in ("kfold", "logo"):
            # Model will be loaded per fold
            model = None
        else:
            # Load model once for all folds (item, group splits)
            checkpoint_path = find_checkpoint_path(
                checkpoint_dir,
                variant_name,
                split_mode,
                fold_idx=None,
            )
            
            if checkpoint_path is None:
                logger.warning(f"Checkpoint not found for variant {variant_name}")
                logger.warning(f"  Searched in: {checkpoint_dir}")
                logger.warning(f"Skipping variant {variant_name}")
                continue
            
            logger.info(f"Loading model: {checkpoint_path}")
            model = load_model_with_fusion_type(
                config_path,
                checkpoint_path,
                fusion_type=fusion_type,
                device=device,
            )
            logger.info("Model loaded successfully")
        
        # Create evaluator
        evaluator = StressTestEvaluator(device, variant_cfg)
        
        # Evaluate each fold
        for fold_idx, (train_subset, val_subset) in enumerate(folds):
            logger.info(f"  Fold {fold_idx + 1}/{n_folds}")
            
            # For k-fold and logo, load model for this specific fold
            if split_mode in ("kfold", "logo"):
                # For logo mode, extract the held-out group ID from the validation dataset
                held_out_group = None
                if split_mode == "logo":
                    # Get the group ID from the validation dataset
                    # For logo, val_subset should be a GroupDynamicsDataset with include_groups
                    if hasattr(val_subset, "group_ids") and len(val_subset.group_ids) == 1:
                        held_out_group = val_subset.group_ids[0]
                    elif hasattr(val_subset, "dataset") and hasattr(val_subset.dataset, "group_ids"):
                        # Handle Subset wrapper
                        held_out_group = val_subset.dataset.group_ids[0] if len(val_subset.dataset.group_ids) == 1 else None
                
                checkpoint_path = find_checkpoint_path(
                    checkpoint_dir,
                    variant_name,
                    split_mode,
                    fold_idx=fold_idx,
                    held_out_group=held_out_group,
                )
                
                if checkpoint_path is None:
                    logger.warning(f"Checkpoint not found for variant {variant_name}, fold {fold_idx}")
                    if split_mode == "kfold":
                        logger.warning(f"  Searched in: {os.path.join(checkpoint_dir, 'kfold', f'fold_{fold_idx:02d}')}")
                    elif split_mode == "logo":
                        search_path = os.path.join(checkpoint_dir, "logo", f"fold_{held_out_group:02d}") if held_out_group else os.path.join(checkpoint_dir, "logo")
                        logger.warning(f"  Searched in: {search_path}")
                    logger.warning(f"Skipping fold {fold_idx + 1} for variant {variant_name}")
                    continue
                
                logger.info(f"    Loading model for fold {fold_idx + 1}: {checkpoint_path}")
                model = load_model_with_fusion_type(
                    config_path,
                    checkpoint_path,
                    fusion_type=fusion_type,
                    device=device,
                )
                logger.info(f"    Model loaded successfully for fold {fold_idx + 1}")
            
            # Build data loader for this fold (reused for all scenarios and sweeps)
            logger.info(f"    Building data loader for fold {fold_idx + 1}...")
            val_loader = build_dataloader(val_subset, variant_cfg, shuffle=False, is_val=True)
            
            # Check if we have baseline cached
            baseline_key = (variant_name, fold_idx)
            baseline_metrics = None
            
            # Evaluate all corruption scenarios
            for corr_type, param_sweep in CORRUPTION_SWEEPS.items():
                for param in param_sweep:
                    # Check if this is baseline
                    if is_baseline(corr_type, param):
                        # Use cached baseline if available
                        if baseline_key in baseline_cache:
                            baseline_metrics = baseline_cache[baseline_key]
                            logger.info(f"    Reusing baseline for {corr_type}={param}")
                        else:
                            # Compute baseline once
                            logger.info(f"    Computing baseline (corruption={corr_type}, param={param})")
                            baseline_metrics = evaluator.evaluate(
                                model, val_loader,
                                corruption_type=None,
                                corruption_param=None,
                            )
                            baseline_cache[baseline_key] = baseline_metrics
                        
                        # Add baseline result for this specific corruption type
                        collector.add_result(
                            variant_name,
                            corr_type,
                            param,
                            fold_idx,
                            baseline_metrics,
                        )
                    else:
                        # Evaluate with corruption
                        logger.info(f"    Evaluating {corr_type}={param}")
                        metrics = evaluator.evaluate(
                            model, val_loader,
                            corruption_type=corr_type,
                            corruption_param=param,
                        )
                        collector.add_result(
                            variant_name,
                            corr_type,
                            param,
                            fold_idx,
                            metrics,
                        )
                    
                    total_runs += 1
            
            # Cleanup data loader and model (for k-fold/logo) for this fold before moving to next fold
            logger.info(f"    Cleaning up data loader for fold {fold_idx + 1}...")
            del val_loader
            if split_mode in ("kfold", "logo"):
                # For k-fold/logo, we need to free the model memory before loading the next fold's model
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info("")
    
    # Export results
    logger.info("=" * 80)
    logger.info("Exporting results...")
    per_fold_path, per_construct_path, all_constructs_path = collector.export_csv(
        output_dir,
        filename_prefix="stress_test_results"
    )
    logger.info(f"Per-fold results: {per_fold_path}")
    logger.info(f"Per-construct aggregated results: {per_construct_path}")
    logger.info(f"All-constructs aggregated results: {all_constructs_path}")
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("Evaluation Complete")
    logger.info("=" * 80)
    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    if total_runs > 0:
        logger.info(f"Average time per run: {elapsed/total_runs:.2f}s")
    logger.info("")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run experiments evaluation")
    parser.add_argument("--configs", type=str, nargs="+", required=True,
                       help="Paths to config YAML files (one per variant)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save results")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                       help="Variants to evaluate (default: all configs). "
                            "If provided, filters configs to only those matching the variants.")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, default: auto)")
    
    args = parser.parse_args()
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    run_stress_test(
        config_paths=args.configs,
        output_dir=args.output_dir,
        variants=args.variants,
        device=device,
    )

