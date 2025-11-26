from __future__ import annotations

import os
import sys
import yaml
import glob
import re
import math
import logging
import argparse
import gc
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import models.builders as build
from data.dataset import GroupDynamicsDataset
from data.collate import collate_fn
from data.transforms import StandardizeTransform
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from experiments.ood_adaptation.data_sampler import create_finetune_test_split, create_zero_shot_test
from experiments.ood_adaptation.finetuner import FineTuner
from experiments.ood_adaptation.metrics_collector import OODAdaptationMetricsCollector


# ----------------------------- Logging -----------------------------

def setup_logging(output_dir: str, log_name: str = "ood_adaptation") -> logging.Logger:
    """Set up logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"{log_name}.log")
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ----------------------------- Utilities -----------------------------

def discover_group_ids(root_dir: str, modalities: List[str]) -> List[int]:
    """Discover group IDs from dataset."""
    example_mod = modalities[0]
    pattern_csv = os.path.join(root_dir, example_mod, "Group_*.csv")
    pattern_json = os.path.join(root_dir, example_mod, "Group_*.json")
    
    files = glob.glob(pattern_csv) + glob.glob(pattern_json)
    group_ids = set()
    
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"Group_(\d+)\.(csv|json)$", base)
        if m:
            group_ids.add(int(m.group(1)))
    
    return sorted(group_ids)


# ----------------------------- Reproducibility -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


# ----------------------------- Checkpoint helpers -----------------------------

def find_logo_checkpoint(checkpoint_base_dir: str, fold_idx: int) -> str:
    """Find checkpoint for a specific LOGO fold."""
    fold_dir = os.path.join(checkpoint_base_dir, "logo", f"fold_{fold_idx:02d}")
    
    # Try different checkpoint names
    for name in ["best.pth", "latest.pth", "epoch_100.pth"]:
        path = os.path.join(fold_dir, name)
        if os.path.isfile(path):
            return path
    
    # Try epoch_*.pth files
    epoch_files = glob.glob(os.path.join(fold_dir, "epoch_*.pth"))
    if epoch_files:
        # Get latest epoch
        def extract_epoch(p):
            m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(p))
            return int(m.group(1)) if m else -1
        epoch_files.sort(key=extract_epoch)
        return epoch_files[-1]
    
    raise FileNotFoundError(f"No checkpoint found in {fold_dir}")


def load_base_model(
    cfg: Dict,
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """Load base model from checkpoint."""
    model = build.model(cfg).to(device)
    
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    
    return model


def build_val_dataset(
    cfg: Dict,
    held_out_group: int
) -> GroupDynamicsDataset:
    """Build validation dataset for a specific held-out group."""
    args = dict(cfg["dataset"]["args"])
    
    # Load normalization stats
    norm_stats = cfg["dataset"].get("stats_dir")
    if isinstance(norm_stats, str):
        with open(norm_stats, "r") as f:
            norm_stats = yaml.safe_load(f)
    
    if norm_stats:
        transform = StandardizeTransform(norm_stats)
        args["transforms"] = transform
    
    # Include only the held-out group
    args["include_groups"] = [held_out_group]

    # Constrain file-backed cache size to avoid exhausting OS file descriptors.
    workers_cfg = cfg["training"].get("num_workers", 4)
    try:
        num_workers = int(workers_cfg)
    except (TypeError, ValueError):
        num_workers = 0
    num_workers = max(1, num_workers)
    desired_cap = int(args.get("npy_cache_cap", 256))
    max_fd_budget = int(cfg["training"].get("max_fd_budget", 512))
    safe_cap = max(16, min(desired_cap, max_fd_budget // (num_workers + 1)))
    args["npy_cache_cap"] = safe_cap
    
    return GroupDynamicsDataset(**args)


def create_dataloaders(
    cfg: Dict,
    train_dataset,
    val_dataset,
    batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation."""
    num_workers = cfg["training"].get("num_workers", 4)
    prefetch_factor = cfg["training"].get("prefetch_factor", 4)
    
    if batch_size is None:
        batch_size = cfg["training"]["batch_size"]
    
    pin_memory_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else ""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        pin_memory_device=pin_memory_device,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(0),
    )
    
    val_batch_size = cfg["training"].get("val_batch_size", batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        pin_memory_device=pin_memory_device,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(1),
    )
    
    return train_loader, val_loader


# ----------------------------- Experiment 1: Data Portion Sweep -----------------------------

def run_data_portion_sweep(
    cfg: Dict,
    checkpoint_base_dir: str,
    output_dir: str,
    device: torch.device,
    logger: logging.Logger,
    data_proportions: List[float] = None,
    full_finetune_epochs: int = 20,
    finetune_lr_divisor: float = 1.0,
    seed: int = 42,
) -> OODAdaptationMetricsCollector:
    """
    Run Experiment 1: Data-portion sweep with fixed epochs.
    
    For each LOGO fold:
        - Load base model checkpoint
        - For each data proportion:
            - Sample that proportion from left-out group for fine-tuning
            - Use remainder as test set
            - Fine-tune for fixed number of epochs (unfreeze all layers)
            - Evaluate after each epoch
    
    Args:
        cfg: Configuration dictionary
        checkpoint_base_dir: Base directory containing LOGO checkpoints
        output_dir: Directory to save results
        device: Device to run on
        logger: Logger instance
        data_proportions: List of proportions to sweep (default: [0, 0.05, 0.10, 0.15, 0.20, 0.25])
        full_finetune_epochs: Number of epochs for full fine-tuning (all layers) (default: 20)
        finetune_lr_divisor: Divide base LR by this factor for fine-tuning (default: 1.0)
        seed: Random seed for data sampling
    
    Returns:
        Metrics collector with all results
    """
    if data_proportions is None:
        data_proportions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    # Discover all groups
    root_dir = cfg["dataset"]["args"]["root_dir"]
    modalities = cfg["dataset"]["args"]["modalities"]
    all_groups = discover_group_ids(root_dir, modalities)
    
    logger.info("=" * 80)
    logger.info("EXPERIMENT 1: Data-Portion Sweep")
    logger.info("=" * 80)
    logger.info(f"Number of LOGO folds: {len(all_groups)}")
    logger.info(f"Groups (held-out): {all_groups}")
    logger.info(f"Data proportions: {data_proportions}")
    logger.info(f"Full fine-tuning epochs: {full_finetune_epochs}")
    logger.info(f"Base LR: {cfg['training']['learning_rate']}")
    logger.info(f"Fine-tuning LR: {cfg['training']['learning_rate'] / finetune_lr_divisor}")
    logger.info(f"Seed: {seed}")
    logger.info("")
    
    # Initialize metrics collector
    collector = OODAdaptationMetricsCollector()
    collector.metadata["n_folds"] = len(all_groups)
    collector.metadata["base_model"] = cfg["model"]["name"]
    collector.metadata["data_proportions"] = data_proportions
    collector.metadata["full_finetune_epochs"] = full_finetune_epochs
    
    # Iterate over all folds
    for fold_idx, held_out_group in enumerate(all_groups):
        logger.info("=" * 80)
        logger.info(f"FOLD {fold_idx + 1}/{len(all_groups)}: Held-out Group {held_out_group}")
        logger.info("=" * 80)
        
        # Find and load base checkpoint
        try:
            checkpoint_path = find_logo_checkpoint(checkpoint_base_dir, held_out_group)
            logger.info(f"Loading checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            logger.error(f"Checkpoint not found: {e}")
            logger.warning(f"Skipping fold {fold_idx}")
            continue
        
        # Build dataset for held-out group
        logger.info(f"Building dataset for group {held_out_group}...")
        full_val_dataset = build_val_dataset(cfg, held_out_group)
        logger.info(f"Total samples in held-out group: {len(full_val_dataset)}")
        
        # Iterate over data proportions
        for proportion in data_proportions:
            logger.info("-" * 80)
            logger.info(f"Data proportion: {proportion * 100:.1f}%")
            
            if proportion == 0.0:
                # Zero-shot: no fine-tuning, evaluate base model on full dataset
                logger.info("Zero-shot evaluation (no fine-tuning)")
                
                # Load base model
                base_model = load_base_model(cfg, checkpoint_path, device)
                base_model.eval()
                
                # Create test loader
                test_dataset = create_zero_shot_test(full_val_dataset)
                _, test_loader = create_dataloaders(cfg, test_dataset, test_dataset)
                
                # Evaluate
                from engine.validator import Validator
                from engine.utils import BuildAutocastKWargs
                
                autocast_kwargs = BuildAutocastKWargs(cfg, device)
                validator = Validator(device, autocast_kwargs, cfg.get("dataset").get("grp_as_ind", False))
                
                with torch.inference_mode():
                    metrics, val_loss = validator.run(base_model, test_loader)
                
                # Store results (epoch 0 for zero-shot)
                collector.add_result(
                    fold_idx=held_out_group,
                    proportion=proportion,
                    epoch=0,
                    metrics=metrics,
                    train_loss=None,
                )
                
                if val_loss is not None:
                    logger.info(f"  Val Loss: {val_loss:.4f}")
                else:
                    logger.info(f"  Val Loss: None")
                
                # Log F1 macro for all labels (zero-shot)
                if metrics:
                    for split_name in ["individual", "group"]:
                        if split_name in metrics:
                            for head_name, head_metrics in metrics[split_name].items():
                                if "f1_macro" in head_metrics:
                                    f1_value = head_metrics["f1_macro"]
                                    logger.info(f"  {split_name.capitalize()} {head_name} F1 Macro: {f1_value:.4f}")
                
                # Cleanup
                del base_model, test_loader, test_dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            else:
                # Fine-tuning with data proportion
                n_finetune = int(len(full_val_dataset) * proportion)
                n_test = len(full_val_dataset) - n_finetune
                logger.info(f"Fine-tuning samples: {n_finetune}, Test samples: {n_test}")
                
                # Split data
                finetune_dataset, test_dataset = create_finetune_test_split(
                    full_val_dataset,
                    finetune_proportion=proportion,
                    seed=seed,
                    stratify=True,
                )
                
                # Create data loaders
                train_loader, val_loader = create_dataloaders(cfg, finetune_dataset, test_dataset)
                
                # Load base model (fresh copy for this proportion)
                model = load_base_model(cfg, checkpoint_path, device)
                
                # Create optimizer with reduced LR
                finetune_cfg = dict(cfg)
                finetune_lr = cfg["training"]["learning_rate"] / finetune_lr_divisor
                finetune_cfg["training"]["learning_rate"] = finetune_lr
                
                optimizer = build_optimizer(model, finetune_cfg)
                
                # Create scheduler (ReduceLROnPlateau on training loss)
                scheduler_cfg = {
                    "type": "reduce_on_plateau",
                    "factor": 0.5,
                    "patience": 3,
                    "threshold": 0.1,
                    "mode": "min",
                    "threshold_mode": "rel",
                }
                scheduler = build_scheduler(optimizer, scheduler_cfg)
                
                # Create fine-tuner (unfreeze all layers for Experiment 1)
                finetuner = FineTuner(
                    cfg=cfg,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    logger=logger,
                    writer=None,
                    world_size=1,
                    freeze_backbone=False,  # Unfreeze all layers
                )
                
                # Fine-tune for specified epochs
                for epoch in range(1, full_finetune_epochs + 1):
                    logger.info(f"  Epoch {epoch}/{full_finetune_epochs}")
                    
                    # Train
                    train_loss = finetuner.train_epoch(epoch)
                    if train_loss is not None:
                        logger.info(f"    Train Loss: {train_loss:.4f}")
                        finetuner.step_scheduler_on_epoch(train_loss)
                    else:
                        logger.info(f"    Train Loss: None")
                    
                    # Validate
                    metrics, val_loss = finetuner.validate()
                    if val_loss is not None:
                        logger.info(f"    Val Loss: {val_loss:.4f}")
                    else:
                        logger.info(f"    Val Loss: None")
                    
                    # Log F1 macro for all labels
                    if metrics:
                        for split_name in ["individual", "group"]:
                            if split_name in metrics:
                                for head_name, head_metrics in metrics[split_name].items():
                                    if "f1_macro" in head_metrics:
                                        f1_value = head_metrics["f1_macro"]
                                        logger.info(f"    {split_name.capitalize()} {head_name} F1 Macro: {f1_value:.4f}")
                    
                    # Store results
                    collector.add_result(
                        fold_idx=held_out_group,
                        proportion=proportion,
                        epoch=epoch,
                        metrics=metrics,
                        train_loss=train_loss,
                    )
                
                # Cleanup
                del model, optimizer, scheduler, finetuner, train_loader, val_loader, finetune_dataset, test_dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        logger.info("")
        if hasattr(full_val_dataset, "close"):
            full_val_dataset.close()
        del full_val_dataset
        gc.collect()
    
    # Export results
    logger.info("=" * 80)
    logger.info("Exporting Experiment 1 Results...")
    logger.info("=" * 80)
    
    per_epoch_csv, agg_csv = collector.export_csv(
        output_dir,
        filename_prefix="exp1_data_portion_sweep"
    )
    
    json_path = os.path.join(output_dir, "exp1_data_portion_sweep.json")
    collector.export_json(json_path)
    
    logger.info(f"Per-epoch results: {per_epoch_csv}")
    logger.info(f"Aggregated results: {agg_csv}")
    logger.info(f"JSON results: {json_path}")
    logger.info("")
    
    return collector


# ----------------------------- Experiment 2: Epoch Sweep -----------------------------

def run_epoch_sweep(
    cfg: Dict,
    checkpoint_base_dir: str,
    output_dir: str,
    device: torch.device,
    logger: logging.Logger,
    exp1_collector: Optional[OODAdaptationMetricsCollector] = None,
    epochs_to_extract: List[int] = None,
    seed: int = 42,
) -> OODAdaptationMetricsCollector:
    """
    Run Experiment 2: Epoch sweep analysis (post-processing from Experiment 1).
    
    For each LOGO fold and each data portion used in Experiment 1:
        - Extract metrics at specified epochs [2, 8, 12, 16, 20] from Experiment 1 logs
        - Organize into separate results structure
    
    This is a post-processing step with no new model training.
    
    Args:
        cfg: Configuration dictionary
        checkpoint_base_dir: Base directory containing LOGO checkpoints (unused, kept for compatibility)
        output_dir: Directory to save results
        device: Device to run on (unused, kept for compatibility)
        logger: Logger instance
        exp1_collector: Metrics collector from Experiment 1 (required)
        epochs_to_extract: Epochs to extract from Experiment 1 (default: [2, 8, 12, 16, 20])
        seed: Random seed (unused, kept for compatibility)
    
    Returns:
        Metrics collector with extracted results
    """
    if epochs_to_extract is None:
        epochs_to_extract = [2, 8, 12, 16, 20]
    
    if exp1_collector is None:
        logger.error("Experiment 1 collector is required for Experiment 2")
        raise ValueError("exp1_collector must be provided for Experiment 2")
    
    logger.info("=" * 80)
    logger.info("EXPERIMENT 2: Epoch Sweep Analysis (Post-processing)")
    logger.info("=" * 80)
    logger.info(f"Epochs to extract: {epochs_to_extract}")
    logger.info("This is a post-processing step - no new training will be performed.")
    logger.info("")
    
    # Initialize metrics collector
    collector = OODAdaptationMetricsCollector()
    collector.metadata["n_folds"] = exp1_collector.metadata.get("n_folds")
    collector.metadata["base_model"] = exp1_collector.metadata.get("base_model")
    collector.metadata["epochs"] = epochs_to_extract
    collector.metadata["experiment_type"] = "epoch_sweep_analysis"
    
    # Extract results from Experiment 1 for all folds and all proportions
    total_extracted = 0
    for fold_idx in sorted(exp1_collector.results.keys()):
        logger.info("=" * 80)
        logger.info(f"FOLD: Held-out Group {fold_idx}")
        logger.info("=" * 80)
        
        for proportion in sorted(exp1_collector.results[fold_idx].keys()):
            logger.info(f"  Data proportion: {proportion * 100:.1f}%")
            extracted_count = 0
            
            for epoch in epochs_to_extract:
                if epoch in exp1_collector.results[fold_idx][proportion]:
                    result = exp1_collector.results[fold_idx][proportion][epoch]
                    collector.add_result(
                        fold_idx=fold_idx,
                        proportion=proportion,
                        epoch=epoch,
                        metrics=result["metrics"],
                        train_loss=result.get("train_loss"),
                    )
                    extracted_count += 1
                    total_extracted += 1
            
            logger.info(f"    Extracted {extracted_count}/{len(epochs_to_extract)} epochs")
        
        logger.info("")
    
    logger.info(f"Total results extracted: {total_extracted}")
    logger.info("")
    
    # Export results
    logger.info("=" * 80)
    logger.info("Exporting Experiment 2 Results...")
    logger.info("=" * 80)
    
    per_epoch_csv, agg_csv = collector.export_csv(
        output_dir,
        filename_prefix="exp2_epoch_sweep"
    )
    
    json_path = os.path.join(output_dir, "exp2_epoch_sweep.json")
    collector.export_json(json_path)
    
    logger.info(f"Per-epoch results: {per_epoch_csv}")
    logger.info(f"Aggregated results: {agg_csv}")
    logger.info(f"JSON results: {json_path}")
    logger.info("")
    
    return collector


# ----------------------------- Experiment 3: Frozen Backbone with Item Split -----------------------------

def run_frozen_backbone_item_split(
    cfg: Dict,
    checkpoint_base_dir: str,
    output_dir: str,
    device: torch.device,
    logger: logging.Logger,
    partial_finetune_epochs: int = 10,
    train_ratio: float = 0.8,
    finetune_lr_multiplier: float = 1.0,
    seed: int = 42,
) -> OODAdaptationMetricsCollector:
    """
    Run Experiment 3: Frozen backbone with item-mode 80/20 split.
    
    For each LOGO fold:
        - Load base model checkpoint
        - Take complete left-out group
        - Use item-mode splitting (80/20 train/test)
        - Freeze backbone, train only classifiers
        - Fine-tune for specified epochs
        - Evaluate after each epoch
    
    Args:
        cfg: Configuration dictionary
        checkpoint_base_dir: Base directory containing LOGO checkpoints
        output_dir: Directory to save results
        device: Device to run on
        logger: Logger instance
        partial_finetune_epochs: Number of epochs for partial fine-tuning (frozen backbone) (default: 10)
        train_ratio: Proportion of data for training (default: 0.8)
        finetune_lr_multiplier: Multiply base LR by this factor for fine-tuning (default: 1.0)
        seed: Random seed for data splitting
    
    Returns:
        Metrics collector with all results
    """
    from torch.utils.data import random_split
    
    # Discover all groups
    root_dir = cfg["dataset"]["args"]["root_dir"]
    modalities = cfg["dataset"]["args"]["modalities"]
    all_groups = discover_group_ids(root_dir, modalities)
    
    logger.info("=" * 80)
    logger.info("EXPERIMENT 3: Frozen Backbone with Item-Mode Split")
    logger.info("=" * 80)
    logger.info(f"Number of LOGO folds: {len(all_groups)}")
    logger.info(f"Groups (held-out): {all_groups}")
    logger.info(f"Train/test ratio: {train_ratio * 100:.1f}% / {(1-train_ratio) * 100:.1f}%")
    logger.info(f"Partial fine-tuning epochs: {partial_finetune_epochs}")
    logger.info(f"Base LR: {cfg['training']['learning_rate']}")
    logger.info(f"Fine-tuning LR: {cfg['training']['learning_rate'] * finetune_lr_multiplier}")
    logger.info(f"Seed: {seed}")
    logger.info("")
    
    # Initialize metrics collector
    collector = OODAdaptationMetricsCollector()
    collector.metadata["n_folds"] = len(all_groups)
    collector.metadata["base_model"] = cfg["model"]["name"]
    collector.metadata["train_ratio"] = train_ratio
    collector.metadata["partial_finetune_epochs"] = partial_finetune_epochs
    collector.metadata["experiment_type"] = "frozen_backbone_item_split"
    
    # Iterate over all folds
    for fold_idx, held_out_group in enumerate(all_groups):
        logger.info("=" * 80)
        logger.info(f"FOLD {fold_idx + 1}/{len(all_groups)}: Held-out Group {held_out_group}")
        logger.info("=" * 80)
        
        # Find and load base checkpoint
        try:
            checkpoint_path = find_logo_checkpoint(checkpoint_base_dir, held_out_group)
            logger.info(f"Loading checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            logger.error(f"Checkpoint not found: {e}")
            logger.warning(f"Skipping fold {fold_idx}")
            continue
        
        # Build dataset for held-out group
        logger.info(f"Building dataset for group {held_out_group}...")
        full_val_dataset = build_val_dataset(cfg, held_out_group)
        logger.info(f"Total samples in held-out group: {len(full_val_dataset)}")
        
        # Split data using item-mode (80/20)
        n_total = len(full_val_dataset)
        n_train = int(round(n_total * train_ratio))
        n_test = n_total - n_train
        logger.info(f"Train samples: {n_train}, Test samples: {n_test}")
        
        g = torch.Generator().manual_seed(seed)
        train_subset, test_subset = random_split(full_val_dataset, [n_train, n_test], generator=g)
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(cfg, train_subset, test_subset)
        
        # Load base model
        model = load_base_model(cfg, checkpoint_path, device)
        
        # Create optimizer with fine-tuning LR
        finetune_cfg = dict(cfg)
        finetune_lr = cfg["training"]["learning_rate"] * finetune_lr_multiplier
        finetune_cfg["training"]["learning_rate"] = finetune_lr
        
        optimizer = build_optimizer(model, finetune_cfg)
        
        # Create scheduler (ReduceLROnPlateau on training loss)
        scheduler_cfg = {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 3,
            "threshold": 0.1,
            "mode": "min",
            "threshold_mode": "rel",
        }
        scheduler = build_scheduler(optimizer, scheduler_cfg)
        
        # Create fine-tuner (freeze backbone, train only classifiers)
        finetuner = FineTuner(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            writer=None,
            world_size=1,
            freeze_backbone=True,  # Freeze backbone
        )
        
        # Fine-tune for specified epochs
        for epoch in range(1, partial_finetune_epochs + 1):
            logger.info(f"  Epoch {epoch}/{partial_finetune_epochs}")
            
            # Train
            train_loss = finetuner.train_epoch(epoch)
            if train_loss is not None:
                logger.info(f"    Train Loss: {train_loss:.4f}")
                finetuner.step_scheduler_on_epoch(train_loss)
            else:
                logger.info(f"    Train Loss: None")
            
            # Validate
            metrics, val_loss = finetuner.validate()
            if val_loss is not None:
                logger.info(f"    Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"    Val Loss: None")
            
            # Log F1 macro for all labels
            if metrics:
                for split_name in ["individual", "group"]:
                    if split_name in metrics:
                        for head_name, head_metrics in metrics[split_name].items():
                            if "f1_macro" in head_metrics:
                                f1_value = head_metrics["f1_macro"]
                                logger.info(f"    {split_name.capitalize()} {head_name} F1 Macro: {f1_value:.4f}")
            
            # Store results
            collector.add_result(
                fold_idx=held_out_group,
                proportion=train_ratio,  # Use train_ratio as proportion identifier
                epoch=epoch,
                metrics=metrics,
                train_loss=train_loss,
            )
        
        # Cleanup
        del model, optimizer, scheduler, finetuner, train_loader, val_loader, train_subset, test_subset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(full_val_dataset, "close"):
            full_val_dataset.close()
        del full_val_dataset
        gc.collect()
        
        logger.info("")
    
    # Export results
    logger.info("=" * 80)
    logger.info("Exporting Experiment 3 Results...")
    logger.info("=" * 80)
    
    per_epoch_csv, agg_csv = collector.export_csv(
        output_dir,
        filename_prefix="exp3_frozen_backbone_item_split"
    )
    
    json_path = os.path.join(output_dir, "exp3_frozen_backbone_item_split.json")
    collector.export_json(json_path)
    
    logger.info(f"Per-epoch results: {per_epoch_csv}")
    logger.info(f"Aggregated results: {agg_csv}")
    logger.info(f"JSON results: {json_path}")
    logger.info("")
    
    return collector


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OOD Adaptation Experiments: Transfer learning on LOGO folds"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML used for base LOGO training"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Base directory containing LOGO checkpoints (e.g., checkpoints/SyntalNet)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ood_adaptation",
        help="Directory to save experiment results"
    )
    parser.add_argument(
        "--exp1-only",
        action="store_true",
        help="Run only Experiment 1 (data-portion sweep)"
    )
    parser.add_argument(
        "--exp2-only",
        action="store_true",
        help="Run only Experiment 2 (epoch sweep analysis)"
    )
    parser.add_argument(
        "--exp3-only",
        action="store_true",
        help="Run only Experiment 3 (frozen backbone with item split)"
    )
    parser.add_argument(
        "--data-proportions",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        help="Data proportions for Experiment 1"
    )
    parser.add_argument(
        "--full-finetune-epochs",
        type=int,
        default=20,
        help="Number of epochs for full fine-tuning (all layers) in Experiment 1 (default: 20)"
    )
    parser.add_argument(
        "--epochs-to-extract",
        type=int,
        nargs="+",
        default=[2, 8, 12, 16, 20],
        help="Epochs to extract from Exp1 for Exp2"
    )
    parser.add_argument(
        "--partial-finetune-epochs",
        type=int,
        default=30,
        help="Number of epochs for partial fine-tuning (frozen backbone) in Experiment 3 (default: 30)"
    )
    parser.add_argument(
        "--exp3-train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for Experiment 3 item-mode split (default: 0.8)"
    )
    parser.add_argument(
        "--lr-divisor",
        type=int,
        default=1,
        help="Divide base LR by this factor for fine-tuning (default: 1 for Exp1, unused for Exp3)"
    )
    parser.add_argument(
        "--lr-multiplier",
        type=float,
        default=1.0,
        help="Multiply base LR by this factor for fine-tuning (default: 1.0 for Exp3, unused for Exp1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data sampling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto)"
    )
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    # Load config
    cfg = build.config(args.config)
    
    training_seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(training_seed)
    
    # Override proto_warmup_epochs to 0 for fine-tuning experiments
    if "model" in cfg and "args" in cfg["model"]:
        original_warmup = cfg["model"]["args"].get("proto_warmup_epochs", 0)
        cfg["model"]["args"]["proto_warmup_epochs"] = 0
        if original_warmup != 0:
            logger.info(f"Overriding proto_warmup_epochs from {original_warmup} to 0 for fine-tuning")
    
    # Override warmup_ratio to 0.05 for fine-tuning experiment 1
    if "training" in cfg:
        original_warmup_ratio = cfg["training"].get("warmup_ratio", 0.1)
        cfg["training"]["warmup_ratio"] = 0.05
        if original_warmup_ratio != 0.05:
            logger.info(f"Overriding warmup_ratio from {original_warmup_ratio} to 0.05 for fine-tuning experiment 1")
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 80)
    logger.info("OOD Adaptation Experiments")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Device: {device}")
    logger.info("")
    
    # Run experiments
    exp1_collector = None
    exp2_collector = None
    exp3_collector = None
    
    run_exp1 = not (args.exp2_only or args.exp3_only)
    run_exp2 = not (args.exp1_only or args.exp3_only)
    run_exp3 = not (args.exp1_only or args.exp2_only)
    
    if run_exp1:
        exp1_collector = run_data_portion_sweep(
            cfg=cfg,
            checkpoint_base_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device=device,
            logger=logger,
            data_proportions=args.data_proportions,
            full_finetune_epochs=args.full_finetune_epochs,
            finetune_lr_divisor=args.lr_divisor,
            seed=args.seed,
        )
    
    if run_exp2:
        exp2_collector = run_epoch_sweep(
            cfg=cfg,
            checkpoint_base_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device=device,
            logger=logger,
            exp1_collector=exp1_collector,
            epochs_to_extract=args.epochs_to_extract,
            seed=args.seed,
        )
    
    # Override warmup_ratio to 0.0 for fine-tuning experiment 3
    if "training" in cfg:
        original_warmup_ratio = cfg["training"].get("warmup_ratio", 0.1)
        cfg["training"]["warmup_ratio"] = 0.0
        if original_warmup_ratio != 0.0:
            logger.info(f"Overriding warmup_ratio from {original_warmup_ratio} to 0.0 for fine-tuning experiment 3")

    if run_exp3:
        exp3_collector = run_frozen_backbone_item_split(
            cfg=cfg,
            checkpoint_base_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device=device,
            logger=logger,
            partial_finetune_epochs=args.partial_finetune_epochs,
            train_ratio=args.exp3_train_ratio,
            finetune_lr_multiplier=args.lr_multiplier,
            seed=args.seed,
        )
    
    logger.info("=" * 80)
    logger.info("All Experiments Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved in: {args.output_dir}")
    logger.info("")


if __name__ == "__main__":
    main()

