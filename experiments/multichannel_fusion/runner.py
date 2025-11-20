from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import sys
import time
import logging
import gc
from pathlib import Path
import torch

import models.builders as build
from experiments.multichannel_fusion.model_loader import (
    load_model_with_fusion_type,
    get_fusion_type_from_variant,
    find_checkpoint_path,
)
from experiments.common.dataset_builder import build_datasets, build_dataloader
from experiments.multichannel_fusion.evaluator import StressTestEvaluator
from experiments.common.metrics_collector import MetricsCollector

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


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


def run_stress_test(
    config_path: str,
    checkpoint_dir: str,
    output_dir: str,
    variants: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
):
    """
    Run complete stress test evaluation.
    
    Args:
        config_path: Path to base config YAML
        checkpoint_dir: Directory containing checkpoints for each variant
        output_dir: Directory to save results
        variants: List of variant names to evaluate (default: all)
        device: Device to run on (default: auto-detect)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if variants is None:
        variants = MODEL_VARIANTS
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("=" * 80)
    logger.info("Starting Stress Test Evaluation")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Variants: {variants}")
    logger.info(f"Device: {device}")
    logger.info("")
    
    # Load config
    cfg = build.config(config_path)
    
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
    
    for variant_idx, variant_name in enumerate(variants):
        logger.info("=" * 80)
        logger.info(f"Evaluating variant: {variant_name} ({variant_idx + 1}/{len(variants)})")
        logger.info("=" * 80)
        
        # Get fusion type for this variant
        fusion_type = get_fusion_type_from_variant(variant_name)
        
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
                logger.warning(f"  Searched in: {os.path.join(checkpoint_dir, variant_name)}")
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
        evaluator = StressTestEvaluator(device, cfg)
        
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
                        logger.warning(f"  Searched in: {os.path.join(checkpoint_dir, variant_name, 'kfold', f'fold_{fold_idx:02d}')}")
                    elif split_mode == "logo":
                        search_path = os.path.join(checkpoint_dir, variant_name, "logo", f"fold_{held_out_group:02d}") if held_out_group else os.path.join(checkpoint_dir, variant_name, "logo")
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
            val_loader = build_dataloader(val_subset, cfg, shuffle=False, is_val=True)
            
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
    per_fold_path, aggregated_path = collector.export_csv(
        output_dir,
        filename_prefix="stress_test_results"
    )
    logger.info(f"Per-fold results: {per_fold_path}")
    logger.info(f"Aggregated results: {aggregated_path}")
    
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
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory with checkpoints (subdirs: bscx, proj_only, etc.)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save results")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                       help="Variants to evaluate (default: all)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, default: auto)")
    
    args = parser.parse_args()
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    run_stress_test(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        variants=args.variants,
        device=device,
    )

