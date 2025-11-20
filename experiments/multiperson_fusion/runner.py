from __future__ import annotations
from typing import Dict, List, Optional
import os
import sys
import time
import logging
import json
import subprocess
import yaml
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.multiperson_fusion.metrics_collector import MultipersonFusionMetricsCollector


def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "multiperson_fusion.log")
    
    # Create logger
    logger = logging.getLogger("multiperson_fusion")
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


def get_variant_name_from_config(config_path: str) -> str:
    """
    Extract variant name from config file path.
    E.g., 'configs/EXPT_SoSEX_w-mp-fus_wo-grp-cls.yaml' -> 'w-mp-fus_wo-grp-cls'
    """
    basename = os.path.basename(config_path)
    # Remove .yaml extension
    name = os.path.splitext(basename)[0]
    # Extract variant name (after EXPT_SoSEX_)
    if "EXPT_SoSEX_" in name:
        variant = name.replace("EXPT_SoSEX_", "")
    else:
        # Fallback: use filename without extension
        variant = name
    return variant


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


def run_evaluate(
    config_path: str,
    checkpoint_path: str,
    output_json_path: Optional[str] = None,
    fold_idx: Optional[int] = None,
) -> Dict:
    """
    Run evaluate.py and return the metrics.
    
    Args:
        config_path: Path to config YAML
        checkpoint_path: Path to checkpoint file or directory
        output_json_path: Optional path to save JSON output
        fold_idx: Optional fold index for kfold/logo modes
    
    Returns:
        Metrics dictionary
    """
    cmd = [
        sys.executable,
        "evaluate.py",
        "--config", config_path,
        "--checkpoint", checkpoint_path,
    ]
    
    if output_json_path:
        cmd.extend(["--out", output_json_path])
    
    if fold_idx is not None:
        cmd.extend(["--fold-idx", str(fold_idx)])
    
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    
    # Load metrics from JSON if provided, otherwise parse from stdout
    if output_json_path and os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            metrics = json.load(f)
    else:
        # Try to parse from stdout (fallback)
        # This is less reliable, but works if JSON output wasn't requested
        try:
            # Look for JSON in stdout
            import re
            json_match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
            if json_match:
                metrics = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse metrics from output")
        except:
            raise ValueError(f"Could not parse metrics. stdout: {result.stdout[:500]}")
    
    return metrics


def run_sosex_experiments(
    config_paths: List[str],
    checkpoint_base_dir: Optional[str] = None,
    output_dir: str = "results/multiperson_fusion",
    device: Optional[torch.device] = None,
):
    """
    Run SOSEX experiments on multiple model variants.
    
    Args:
        config_paths: List of paths to config YAML files for each variant
        checkpoint_base_dir: Base directory for checkpoints (if None, extracted from config)
        output_dir: Directory to save results
        device: Device to run on (default: auto-detect)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("=" * 80)
    logger.info("Starting Experiments Evaluation")
    logger.info("=" * 80)
    logger.info(f"Configs: {config_paths}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info("")
    
    # Initialize metrics collector
    collector = SOSEXMetricsCollector()
    
    total_runs = 0
    start_time = time.time()
    
    # Evaluate each variant
    for variant_idx, config_path in enumerate(config_paths):
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, skipping...")
            continue
        
        variant_name = get_variant_name_from_config(config_path)
        logger.info("=" * 80)
        logger.info(f"Evaluating variant: {variant_name} ({variant_idx + 1}/{len(config_paths)})")
        logger.info("=" * 80)
        logger.info(f"Config: {config_path}")
        
        # Get checkpoint directory
        try:
            checkpoint_dir = get_checkpoint_dir_from_config(config_path)
            if checkpoint_base_dir:
                # Use provided base directory, but check if variant subdirectory exists
                variant_checkpoint_dir = os.path.join(checkpoint_base_dir, variant_name)
                if os.path.exists(variant_checkpoint_dir):
                    checkpoint_dir = variant_checkpoint_dir
                else:
                    # Try using the checkpoint_dir from config as-is
                    pass
        except Exception as e:
            logger.error(f"Failed to get checkpoint directory: {e}")
            logger.warning(f"Skipping variant {variant_name}")
            continue
        
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        
        # Load config to determine split mode
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        split_cfg = cfg.get("dataset", {}).get("split", {})
        split_mode = split_cfg.get("mode", "item")
        logger.info(f"Split mode: {split_mode}")
        
        # Determine if we need to evaluate multiple folds
        if split_mode in ("kfold", "logo"):
            # For kfold, get n_folds from config
            if split_mode == "kfold":
                n_folds = int(split_cfg.get("n_folds", 5))
                fold_indices = list(range(n_folds))
            else:
                # For logo, we need to discover groups
                # For now, we'll let evaluate.py handle all folds
                fold_indices = None  # None means evaluate all folds
        else:
            # Single fold evaluation
            fold_indices = [None]
        
        # Run evaluation
        try:
            if fold_indices is None:
                # Evaluate all folds at once (evaluate.py will handle it)
                logger.info(f"Running evaluation for all folds...")
                temp_json = os.path.join(output_dir, f"temp_{variant_name}_metrics.json")
                metrics = run_evaluate(
                    config_path=config_path,
                    checkpoint_path=checkpoint_dir,
                    output_json_path=temp_json,
                    fold_idx=None,
                )
                
                # Process metrics
                if "per_fold" in metrics and "aggregated" in metrics:
                    # Multi-fold results
                    for fold_num, fold_metrics in enumerate(metrics["per_fold"]):
                        collector.add_result(
                            variant_name=variant_name,
                            fold_idx=fold_num,
                            metrics=fold_metrics,
                        )
                        total_runs += 1
                else:
                    # Single fold result
                    collector.add_result(
                        variant_name=variant_name,
                        fold_idx=None,
                        metrics=metrics,
                    )
                    total_runs += 1
                
                # Cleanup temp file
                if os.path.exists(temp_json):
                    os.remove(temp_json)
            else:
                # Evaluate each fold separately
                for fold_idx in fold_indices:
                    logger.info(f"  Evaluating fold {fold_idx if fold_idx is not None else 'single'}...")
                    temp_json = os.path.join(output_dir, f"temp_{variant_name}_fold{fold_idx}_metrics.json")
                    metrics = run_evaluate(
                        config_path=config_path,
                        checkpoint_path=checkpoint_dir,
                        output_json_path=temp_json,
                        fold_idx=fold_idx,
                    )
                    
                    collector.add_result(
                        variant_name=variant_name,
                        fold_idx=fold_idx,
                        metrics=metrics,
                    )
                    total_runs += 1
                    
                    # Cleanup temp file
                    if os.path.exists(temp_json):
                        os.remove(temp_json)
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed for variant {variant_name}: {e}")
            logger.error(f"stdout: {e.stdout[:500] if e.stdout else 'N/A'}")
            logger.error(f"stderr: {e.stderr[:500] if e.stderr else 'N/A'}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error for variant {variant_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
        
        logger.info("")
    
    # Export results
    logger.info("=" * 80)
    logger.info("Exporting results...")
    per_fold_path, aggregated_path = collector.export_csv(
        output_dir,
        filename_prefix="sosex_experiments_results"
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
    parser.add_argument("--configs", type=str, nargs="+", required=True,
                       help="List of config YAML files for each variant")
    parser.add_argument("--checkpoint-base-dir", type=str, default=None,
                       help="Base directory for checkpoints (optional, will use config if not provided)")
    parser.add_argument("--output-dir", type=str, default="results/multiperson_fusion",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, default: auto)")
    
    args = parser.parse_args()
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    run_sosex_experiments(
        config_paths=args.configs,
        checkpoint_base_dir=args.checkpoint_base_dir,
        output_dir=args.output_dir,
        device=device,
    )

