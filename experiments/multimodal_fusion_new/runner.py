"""
Main runner for reliability-switch experiments.

Trains and evaluates multimodal fusion variants on top of frozen SyntalNet backbones.
"""

import os
import sys
import argparse
import torch
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import models.builders as build
from data.dataset import GroupDynamicsDataset
from data.collate import collate_fn
from data.transforms import StandardizeTransform
from torch.utils.data import DataLoader, random_split, Subset
from utils.logger import setup_logger
from utils.optimizer import build_optimizer

from experiments.multimodal_fusion_new.config import (
    load_config,
    save_config,
    get_fold_dirs,
    get_base_checkpoint_path,
)
from experiments.multimodal_fusion_new.model_loader import load_checkpoint_for_reliability_training
from experiments.multimodal_fusion_new.reliability_trainer import ReliabilityTrainer
from experiments.multimodal_fusion_new.evaluator import StressTestEvaluator
from experiments.multimodal_fusion_new.fusion_variants import get_variant_config
from experiments.multimodal_fusion_new.corruptions import ReliabilitySwitchConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reliability-switch experiments for multimodal fusion variants"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "both"],
        default="both",
        help="Run mode: train only, eval only, or both"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to run (if None, runs all folds)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    return parser.parse_args()


def _create_kfold_splits(dataset, n_folds: int, fold_idx: int, seed: int):
    """Create train/val splits for K-fold CV."""
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(n, generator=g).tolist()
    
    fold_size = n // n_folds
    remainder = n % n_folds
    start_idx = fold_idx * fold_size + min(fold_idx, remainder)
    end_idx = start_idx + fold_size + (1 if fold_idx < remainder else 0)
    
    val_indices = all_indices[start_idx:end_idx]
    train_indices = all_indices[:start_idx] + all_indices[end_idx:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    return train_subset, val_subset


def build_dataloaders(base_config, exp_config, fold_idx: int):
    """Build train and val dataloaders for a specific fold."""
    
    # Load dataset args from base SyntalNet config
    dataset_args = base_config["dataset"]["args"]
    
    # Load normalization stats if available
    norm_stats_path = base_config["dataset"].get("stats_dir")
    if norm_stats_path and os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            norm_stats = yaml.safe_load(f)
        transform = StandardizeTransform(norm_stats)
        dataset_args["transforms"] = transform
    
    # Build full dataset
    full_dataset = GroupDynamicsDataset(**dataset_args)
    
    # Split according to cv_mode
    cv_mode = exp_config.get("cv_mode", "kfold")
    n_folds = exp_config.get("n_folds", 5)
    seed = base_config["training"].get("seed", 42)
    
    if cv_mode == "kfold":
        train_dataset, val_dataset = _create_kfold_splits(
            full_dataset, n_folds, fold_idx, seed
        )
    else:
        raise NotImplementedError(f"CV mode {cv_mode} not implemented yet")
    
    # Build dataloaders
    batch_size = base_config["training"].get("batch_size", 8)
    num_workers = base_config["training"].get("num_workers", 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


def run_fold(
    fold_idx: int,
    exp_config: dict,
    base_config: dict,
    mode: str,
    device: torch.device,
):
    """Run training and/or evaluation for a single fold."""
    
    cv_mode = exp_config.get("cv_mode", "kfold")
    fusion_variant = exp_config["fusion_variant"]
    
    # Get directories for this fold
    log_dir, checkpoint_dir, results_dir = get_fold_dirs(
        exp_config["logging"]["log_dir"],
        exp_config["logging"]["checkpoint_dir"],
        exp_config["logging"]["results_dir"],
        fusion_variant,
        fold_idx,
        cv_mode,
    )
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logger
    logger, _ = setup_logger(log_dir, name=f"fold_{fold_idx:02d}")
    logger.info(f"=" * 80)
    logger.info(f"Running fold {fold_idx} | Fusion variant: {fusion_variant}")
    logger.info(f"=" * 80)
    
    # Save experiment config
    save_config(exp_config, os.path.join(log_dir, "experiment_config.yaml"))
    
    # Build dataloaders
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(base_config, exp_config, fold_idx)
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Load base checkpoint and build model
    logger.info("Loading base SyntalNet checkpoint...")
    base_checkpoint_path = get_base_checkpoint_path(
        exp_config["base_checkpoint_dir"],
        fold_idx,
        cv_mode,
    )
    logger.info(f"Base checkpoint: {base_checkpoint_path}")
    
    # Get fusion kwargs
    fusion_kwargs = exp_config.get("fusion_kwargs", {})
    if not fusion_kwargs:
        fusion_kwargs = get_variant_config(fusion_variant)
    
    model, _ = load_checkpoint_for_reliability_training(
        checkpoint_path=base_checkpoint_path,
        config=base_config,
        fusion_variant=fusion_variant,
        fusion_kwargs=fusion_kwargs,
        device=device,
    )
    
    # Training
    if mode in ["train", "both"]:
        logger.info("Starting reliability-switch training...")
        
        # Build optimizer (only for trainable parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        optimizer_cfg = {
            "training": {
                "learning_rate": exp_config["reliability_training"]["learning_rate"],
                "weight_decay": exp_config["reliability_training"]["weight_decay"],
            }
        }
        optimizer = build_optimizer(model, optimizer_cfg)
        
        # Build scheduler
        scheduler_type = exp_config["reliability_training"].get("scheduler", "reduce_on_plateau")
        if scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=exp_config["reliability_training"].get("scheduler_patience", 3),
                factor=exp_config["reliability_training"].get("scheduler_factor", 0.5),
                min_lr=exp_config["reliability_training"].get("min_lr", 1e-6),
            )
        else:
            raise NotImplementedError(f"Scheduler {scheduler_type} not implemented")
        
        # Build corruption config
        corruption_cfg = ReliabilitySwitchConfig.from_dict(exp_config["reliability_training"])
        
        # Build trainer
        trainer = ReliabilityTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            base_config=base_config,
            corruption_config=corruption_cfg,
            logger=logger,
            variant_name=fusion_variant,
        )
        
        # Train
        trainer.train(
            num_epochs=exp_config["reliability_training"]["num_epochs"],
            checkpoint_dir=checkpoint_dir,
            validate_interval=exp_config["reliability_training"]["validate_interval"],
            save_interval=exp_config["reliability_training"]["save_interval"],
        )
        
        logger.info("Training complete.")
    
    # Evaluation
    if mode in ["eval", "both"]:
        logger.info("Starting stress test evaluation...")
        
        # Load best checkpoint if we just trained
        if mode == "both":
            best_ckpt = os.path.join(checkpoint_dir, "best_model.pth")
            if os.path.exists(best_ckpt):
                logger.info(f"Loading best checkpoint: {best_ckpt}")
                ckpt = torch.load(best_ckpt, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
        
        # Build evaluator
        evaluator = StressTestEvaluator(
            model=model,
            loader=val_loader,
            device=device,
            variant_name=fusion_variant,
            logger=logger,
        )
        
        # Evaluate clean performance
        logger.info("Evaluating on clean data...")
        clean_metrics = evaluator.evaluate_clean()
        logger.info(f"Clean F1: {clean_metrics.get('f1_macro', 0.0):.4f}")
        
        # Save clean metrics
        clean_df = pd.DataFrame([{
            'fold': fold_idx,
            'variant': fusion_variant,
            'corruption_type': 'none',
            'corruption_param': 0.0,
            'modality': 'none',
            **{k: v for k, v in clean_metrics.items() if not isinstance(v, (list, dict))}
        }])
        clean_df.to_csv(os.path.join(results_dir, "clean_metrics.csv"), index=False)
        
        # Run stress tests
        logger.info("Running stress tests...")
        stress_results = evaluator.run_full_stress_test(
            corruption_configs=exp_config["stress_test"]["corruption_configs"],
            per_modality=exp_config["stress_test"]["per_modality"],
        )
        
        # Add fold and variant info
        stress_results['fold'] = fold_idx
        stress_results['variant'] = fusion_variant
        
        # Save results
        results_path = os.path.join(results_dir, "stress_test_results.csv")
        stress_results.to_csv(results_path, index=False)
        logger.info(f"Saved stress test results to: {results_path}")
        
        logger.info("Evaluation complete.")
    
    logger.info(f"Fold {fold_idx} complete!")
    logger.info("=" * 80)


def aggregate_results(exp_config: dict):
    """Aggregate results across all folds."""
    
    fusion_variant = exp_config["fusion_variant"]
    cv_mode = exp_config.get("cv_mode", "kfold")
    n_folds = exp_config.get("n_folds", 5)
    
    results_base = Path(exp_config["logging"]["results_dir"]) / cv_mode / fusion_variant.lower()
    
    # Collect all fold results
    all_stress_results = []
    all_clean_results = []
    
    for fold_idx in range(n_folds):
        fold_dir = results_base / f"fold_{fold_idx:02d}"
        
        stress_path = fold_dir / "stress_test_results.csv"
        if stress_path.exists():
            df = pd.read_csv(stress_path)
            all_stress_results.append(df)
        
        clean_path = fold_dir / "clean_metrics.csv"
        if clean_path.exists():
            df = pd.read_csv(clean_path)
            all_clean_results.append(df)
    
    if not all_stress_results:
        print("No results found to aggregate.")
        return
    
    # Combine and save
    stress_combined = pd.concat(all_stress_results, ignore_index=True)
    clean_combined = pd.concat(all_clean_results, ignore_index=True) if all_clean_results else pd.DataFrame()
    
    aggregated_dir = results_base / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    
    stress_combined.to_csv(aggregated_dir / "stress_test_all_folds.csv", index=False)
    if not clean_combined.empty:
        clean_combined.to_csv(aggregated_dir / "clean_metrics_all_folds.csv", index=False)
    
    print(f"Aggregated results saved to: {aggregated_dir}")
    
    # Compute summary statistics
    summary_stats = stress_combined.groupby(
        ['corruption_type', 'corruption_param', 'modality', 'split', 'head', 'metric']
    )['value'].agg(['mean', 'std', 'count']).reset_index()
    
    summary_stats.to_csv(aggregated_dir / "stress_test_summary.csv", index=False)
    print(f"Summary statistics saved to: {aggregated_dir / 'stress_test_summary.csv'}")


def main():
    args = parse_args()
    
    # Load config
    exp_config = load_config(args.config)
    
    # Load base SyntalNet config
    base_config = build.config(exp_config["base_config_path"])
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine folds to run
    cv_mode = exp_config.get("cv_mode", "kfold")
    n_folds = exp_config.get("n_folds", 5)
    
    if args.fold is not None:
        folds_to_run = [args.fold]
    else:
        folds_to_run = list(range(n_folds))
    
    # Run each fold
    for fold_idx in folds_to_run:
        try:
            run_fold(
                fold_idx=fold_idx,
                exp_config=exp_config,
                base_config=base_config,
                mode=args.mode,
                device=device,
            )
        except Exception as e:
            print(f"Error in fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate results if we ran all folds
    if args.fold is None and args.mode in ["eval", "both"]:
        print("\nAggregating results across folds...")
        aggregate_results(exp_config)


if __name__ == "__main__":
    main()

