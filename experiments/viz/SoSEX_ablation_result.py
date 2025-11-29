from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------- Constants -----------------------------

# Model variation identifiers
MODEL_VARIANTS = {
    "no_sosex_no_grphead": {
        "label": r"$\mathbf{-}$SoSE-X, $\mathbf{-}$Group Head",
        "short_label": "-SoSEX,-GrpHead",
        "sosex": False,
        "grp_head": False,
        "default_config": "EXPT_SoSEX_wo-mp-fus_wo-grp-cls.yaml",
    },
    "no_sosex_with_grphead": {
        "label": r"$\mathbf{-}$SoSE-X, $\mathbf{+}$Group Head",
        "short_label": "-SoSEX,+GrpHead",
        "sosex": False,
        "grp_head": True,
        "default_config": "EXPT_SoSEX_wo-mp-fus_w-grp-cls.yaml",
    },
    "with_sosex_no_grphead": {
        "label": r"$\mathbf{+}$SoSE-X, $\mathbf{-}$Group Head",
        "short_label": "+SoSEX,-GrpHead",
        "sosex": True,
        "grp_head": False,
        "default_config": "EXPT_SoSEX_w-mp-fus_wo-grp-cls.yaml",
    },
    "with_sosex_with_grphead": {
        "label": r"$\mathbf{+}$SoSE-X, $\mathbf{+}$Group Head",
        "short_label": "+SoSEX,+GrpHead",
        "sosex": True,
        "grp_head": True,
        "default_config": "SyntalNet.yaml",
    },
}

# Ordered list for visualization
MODEL_ORDER = [
    "no_sosex_no_grphead",
    "no_sosex_with_grphead",
    "with_sosex_no_grphead",
    "with_sosex_with_grphead",
]

# Label constructs in display order
LABEL_CONSTRUCTS = [
    {"name": "Group\nConfidence", "key": "Confidence", "type": "group"},
    {"name": "Group\nSynchrony", "key": "Synchrony", "type": "group"},
    {"name": "Group\nTransition", "key": "Transition", "type": "group"},
    {"name": "Individual\nEngagement", "key": "Engagement", "type": "individual"},
    {"name": "Individual\nLead", "key": "Lead", "type": "individual"},
]

# Metrics to extract
METRICS_OF_INTEREST = {
    "AUPRC": "auprc_macro",
    "AUROC": "auroc_macro",
    "Accuracy": "accuracy",
    "F1 Score": "f1_macro",
}

# Aggregation statistics to save
AGGREGATION_STATS = ["mean", "std", "min", "max", "median", "q1", "q3", "n_folds"]


# ----------------------------- Argument Parsing -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SoSEX Ablation Experiment Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation for all model variants
  python SoSEX_ablation_result.py --run-eval \\
      --no-sosex-no-grphead-config configs/EXPT_SoSEX_wo-mp-fus_wo-grp-cls.yaml \\
      --no-sosex-no-grphead-ckpt checkpoints/EXPT_SoSEX_wo-mp-fus_wo-grp-cls \\
      --no-sosex-with-grphead-config configs/EXPT_SoSEX_wo-mp-fus_w-grp-cls.yaml \\
      --no-sosex-with-grphead-ckpt checkpoints/EXPT_SoSEX_wo-mp-fus_w-grp-cls \\
      --with-sosex-no-grphead-config configs/EXPT_SoSEX_w-mp-fus_wo-grp-cls.yaml \\
      --with-sosex-no-grphead-ckpt checkpoints/EXPT_SoSEX_w-mp-fus_wo-grp-cls \\
      --with-sosex-with-grphead-config configs/SyntalNet.yaml \\
      --with-sosex-with-grphead-ckpt checkpoints/SyntalNet \\
      --output-dir results/sosex_ablation

  # Use pre-saved CSV files
  python SoSEX_ablation_result.py --use-saved --input-dir results/sosex_ablation
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--run-eval",
        action="store_true",
        help="Run evaluation on provided config/checkpoint pairs"
    )
    mode_group.add_argument(
        "--use-saved",
        action="store_true",
        help="Use pre-saved CSV files instead of running evaluation"
    )
    
    # Output/Input directories
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/viz_results/sosex",
        help="Directory to save CSV results and figures (default: ./experiments/viz_results/sosex)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing pre-saved CSV files (default: same as --output-dir)"
    )
    
    # Config/checkpoint pairs for each model variant
    for variant_key in MODEL_ORDER:
        variant_info = MODEL_VARIANTS[variant_key]
        arg_prefix = variant_key.replace("_", "-")
        
        parser.add_argument(
            f"--{arg_prefix}-config",
            type=str,
            default=None,
            help=f"Config file for {variant_info['short_label']} model"
        )
        parser.add_argument(
            f"--{arg_prefix}-ckpt",
            type=str,
            default=None,
            help=f"Checkpoint path for {variant_info['short_label']} model"
        )
    
    # Evaluation options
    parser.add_argument(
        "--fold-idx",
        type=int,
        default=None,
        help="Specific fold to evaluate (for kfold/logo). If not specified, evaluates all folds."
    )
    
    # Visualization options
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't show the plot (only save to file)"
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="sosex_ablation",
        help="Base name for output figure files (default: sosex_ablation)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=list(METRICS_OF_INTEREST.keys()),
        default=list(METRICS_OF_INTEREST.keys()),
        help="Metrics to visualize (default: all metrics). Choices: " + ", ".join(METRICS_OF_INTEREST.keys())
    )
    
    return parser.parse_args()


# ----------------------------- Evaluation Runner -----------------------------

def run_evaluation(config_path: str, checkpoint_path: str, fold_idx: Optional[int] = None) -> dict:
    """
    Run evaluation using the evaluate.py module.
    
    Args:
        config_path: Path to the config YAML file
        checkpoint_path: Path to checkpoint file or directory
        fold_idx: Optional fold index for kfold/logo evaluation
    
    Returns:
        Dictionary containing evaluation metrics
    """
    import yaml
    import torch
    import random
    
    import models.builders as build
    from data.dataset import GroupDynamicsDataset
    from data.collate import collate_fn
    from data.transforms import StandardizeTransform
    from engine.validator import Validator
    from engine.utils import BuildAutocastKWargs
    from torch.utils.data import DataLoader, random_split, Subset
    
    # Load config
    cfg = build.config(config_path)
    
    # Set seed for reproducibility
    seed = int(cfg.get("training", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model uses grp_as_ind
    grp_as_ind = cfg["dataset"].get("grp_as_ind", False)
    
    # Build validation datasets
    from evaluate import build_val_datasets, find_checkpoint_path
    
    split_cfg = cfg["dataset"].get("split", {"mode": "item", "ratio": 0.2})
    split_mode = split_cfg.get("mode", "item")
    
    print(f"  Building validation datasets (split mode: {split_mode})...")
    val_folds = build_val_datasets(cfg, fold_idx=fold_idx)
    n_folds = len(val_folds)
    print(f"  Created {n_folds} fold(s) for evaluation")
    
    # DataLoader settings
    val_bs = int(cfg["training"].get("val_batch_size", cfg["training"].get("batch_size", 8)))
    num_workers = int(cfg["training"].get("num_workers", 4))
    prefetch_factor = cfg["training"].get("prefetch_factor", 4)
    pin_memory_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else ""
    
    def _worker_init_fn(worker_id: int):
        base_seed = torch.initial_seed() % 2**32
        np.random.seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)
    
    # Validation setup
    autocast_kwargs = BuildAutocastKWargs(cfg, device)
    validator = Validator(device=device, autocast_kwargs=autocast_kwargs, grp_as_ind=grp_as_ind)
    
    all_fold_metrics = []
    
    # Evaluate each fold
    for fold_num, (val_dataset, fold_identifier) in enumerate(val_folds):
        fold_label = f"fold {fold_identifier}" if fold_identifier is not None else "validation set"
        if n_folds > 1:
            print(f"  Evaluating {fold_label} ({fold_num + 1}/{n_folds})...")
        
        # Build data loader for this fold
        loader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=torch.Generator().manual_seed(1),
            pin_memory_device=pin_memory_device,
        )
        
        # Load model checkpoint for this fold
        model = build.model(cfg).to(device)
        
        # Find checkpoint path for this fold
        if n_folds > 1 and split_mode in ("kfold", "logo"):
            try:
                ckpt_path = find_checkpoint_path(
                    checkpoint_path,
                    split_mode,
                    fold_idx=fold_identifier if split_mode == "kfold" else None,
                    held_out_group=fold_identifier if split_mode == "logo" else None
                )
            except (FileNotFoundError, ValueError):
                ckpt_path = checkpoint_path
                if n_folds > 1:
                    print(f"  Warning: Fold-specific checkpoint not found, using: {ckpt_path}")
        else:
            ckpt_path = checkpoint_path
        
        state = torch.load(ckpt_path, map_location="cpu")
        
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if missing or unexpected:
            print(f"  [LD] missing: {missing}")
            print(f"  [LD] unexpected: {unexpected}")
        assert len(missing) == 0 and len(unexpected) == 0, "State mismatch!"
        model.eval()
        
        # Run evaluation
        with torch.inference_mode():
            metrics, _ = validator.run(model, loader)
            all_fold_metrics.append(metrics)
        
        # Cleanup
        del loader
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Aggregate results if multiple folds
    if n_folds > 1:
        print(f"  Aggregating {n_folds} folds...")
        from evaluate import aggregate_metrics
        aggregated_metrics = aggregate_metrics(all_fold_metrics)
        return {
            "per_fold": all_fold_metrics,
            "aggregated": aggregated_metrics,
            "n_folds": n_folds,
            "split_mode": split_mode,
            "grp_as_ind": grp_as_ind,
        }
    else:
        return {
            "metrics": all_fold_metrics[0],
            "n_folds": 1,
            "split_mode": split_mode,
            "grp_as_ind": grp_as_ind,
        }


def extract_metrics_for_labels(
    eval_result: dict,
    has_group_head: bool
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Extract metrics for each label construct from evaluation results.
    
    Args:
        eval_result: Dictionary containing evaluation results
        has_group_head: Whether the model has a group head
    
    Returns:
        Dictionary mapping label names to metric values with all aggregation stats
        {
            "Confidence": {
                "AUPRC": {"mean": 0.5, "std": 0.02, "min": 0.45, "max": 0.55, ...},
                "AUROC": {"mean": 0.7, ...},
                ...
            },
            "Synchrony": {...},
            ...
        }
    """
    # Determine if we have aggregated results or single fold
    if "aggregated" in eval_result:
        metrics = eval_result["aggregated"]
        is_aggregated = True
    else:
        metrics = eval_result.get("metrics", eval_result)
        is_aggregated = False
    
    grp_as_ind = eval_result.get("grp_as_ind", not has_group_head)
    
    extracted = {}
    
    for label_info in LABEL_CONSTRUCTS:
        label_key = label_info["key"]
        label_type = label_info["type"]
        
        # Determine where to look for this label's metrics
        if label_type == "individual":
            # Individual labels are always in "individual" split
            split = "individual"
        else:
            # Group labels: in "group" if has_group_head, else in "individual" (grp_as_ind)
            if has_group_head and not grp_as_ind:
                split = "group"
            else:
                split = "individual"
        
        # Try to find the label in the metrics
        label_metrics = {}
        
        if split in metrics and label_key in metrics[split]:
            label_data = metrics[split][label_key]
            
            for display_name, metric_key in METRICS_OF_INTEREST.items():
                if metric_key in label_data:
                    value = label_data[metric_key]
                    # Handle aggregated metrics (dict with all stats)
                    if isinstance(value, dict) and "mean" in value:
                        # Extract all available stats
                        label_metrics[display_name] = {
                            stat: float(value[stat]) 
                            for stat in AGGREGATION_STATS 
                            if stat in value
                        }
                    else:
                        # Single fold - wrap in dict with just "mean"
                        label_metrics[display_name] = {"mean": float(value)}
        
        if label_metrics:
            extracted[label_key] = label_metrics
        else:
            print(f"  Warning: Could not find metrics for {label_key} in {split} split")
    
    return extracted


def save_metrics_to_csv(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    variant_key: str,
    output_dir: str
) -> str:
    """
    Save extracted metrics to a CSV file with all aggregation statistics.
    
    Args:
        metrics: Dictionary mapping label names to metric values with stats
                 {label_key: {metric_name: {stat_name: value}}}
        variant_key: Model variant identifier
        output_dir: Directory to save the CSV file
    
    Returns:
        Path to the saved CSV file
        
    CSV Format:
        Label, AUPRC_mean, AUPRC_std, AUPRC_min, ..., AUROC_mean, AUROC_std, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame with flattened columns
    rows = []
    for label_key, label_metrics in metrics.items():
        row = {"Label": label_key}
        for metric_name, stat_values in label_metrics.items():
            for stat_name, value in stat_values.items():
                col_name = f"{metric_name}_{stat_name}"
                row[col_name] = value
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns: Label first, then sorted metric_stat columns
    cols = ["Label"]
    metric_stat_cols = [c for c in df.columns if c != "Label"]
    # Sort by metric name, then by stat order
    def sort_key(col):
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            metric, stat = parts
            stat_order = AGGREGATION_STATS.index(stat) if stat in AGGREGATION_STATS else 999
            return (metric, stat_order)
        return (col, 999)
    metric_stat_cols.sort(key=sort_key)
    cols.extend(metric_stat_cols)
    df = df[cols]
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"{variant_key}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved metrics to: {csv_path}")
    
    return csv_path


def load_metrics_from_csv(variant_key: str, input_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load metrics from a CSV file with all aggregation statistics.
    
    Args:
        variant_key: Model variant identifier
        input_dir: Directory containing the CSV file
    
    Returns:
        Dictionary mapping label names to metric values with stats
        {label_key: {metric_name: {stat_name: value}}}
    """
    csv_path = os.path.join(input_dir, f"{variant_key}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    metrics = {}
    for _, row in df.iterrows():
        label_key = row["Label"]
        label_metrics = {}
        
        for metric_name in METRICS_OF_INTEREST.keys():
            stat_values = {}
            for stat_name in AGGREGATION_STATS:
                col_name = f"{metric_name}_{stat_name}"
                if col_name in row and pd.notna(row[col_name]):
                    stat_values[stat_name] = float(row[col_name])
            
            # Fallback: check for old format (just metric name without stat suffix)
            if not stat_values and metric_name in row and pd.notna(row[metric_name]):
                stat_values["mean"] = float(row[metric_name])
            
            if stat_values:
                label_metrics[metric_name] = stat_values
        
        metrics[label_key] = label_metrics
    
    return metrics


# ----------------------------- Visualization -----------------------------

def build_metrics_data(all_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> Dict[str, np.ndarray]:
    """
    Build numpy arrays for visualization from extracted metrics.
    Uses the 'mean' value for visualization.
    
    Args:
        all_metrics: Dictionary mapping variant keys to label metrics with stats
                     {variant_key: {label_key: {metric_name: {stat_name: value}}}}
    
    Returns:
        Dictionary mapping metric names to numpy arrays
        {metric_name: array of shape (n_labels, n_variants)}
    """
    n_labels = len(LABEL_CONSTRUCTS)
    n_variants = len(MODEL_ORDER)
    
    metrics_data = {}
    for metric_name in METRICS_OF_INTEREST.keys():
        metrics_data[metric_name] = np.zeros((n_labels, n_variants))
    
    for var_idx, variant_key in enumerate(MODEL_ORDER):
        if variant_key not in all_metrics:
            continue
        
        variant_metrics = all_metrics[variant_key]
        
        for label_idx, label_info in enumerate(LABEL_CONSTRUCTS):
            label_key = label_info["key"]
            
            if label_key not in variant_metrics:
                continue
            
            label_metrics = variant_metrics[label_key]
            
            for metric_name in METRICS_OF_INTEREST.keys():
                if metric_name in label_metrics:
                    stat_values = label_metrics[metric_name]
                    # Use mean value for visualization
                    if isinstance(stat_values, dict) and "mean" in stat_values:
                        metrics_data[metric_name][label_idx, var_idx] = stat_values["mean"]
                    elif isinstance(stat_values, (int, float)):
                        # Backward compatibility with old format
                        metrics_data[metric_name][label_idx, var_idx] = stat_values
    
    return metrics_data


def visualize_ablation(
    metrics_data: Dict[str, np.ndarray],
    all_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    selected_metrics: List[str],
    output_dir: str,
    figure_name: str = "sosex_ablation",
    show: bool = True
):
    """
    Create the ablation visualization figure.
    
    Args:
        metrics_data: Dictionary mapping metric names to numpy arrays (mean values)
        all_metrics: Dictionary mapping variant keys to label metrics with all stats
                     {variant_key: {label_key: {metric_name: {stat_name: value}}}}
        selected_metrics: List of metric names to visualize
        output_dir: Directory to save the figure
        figure_name: Base name for output figure files
        show: Whether to display the figure
    """
    # Filter metrics_data to only include selected metrics
    filtered_metrics_data = {k: v for k, v in metrics_data.items() if k in selected_metrics}
    
    # Get labels and legend labels
    labels = [lc["name"] for lc in LABEL_CONSTRUCTS]
    legend_labels = [MODEL_VARIANTS[vk]["label"] for vk in MODEL_ORDER]
    
    # Style settings
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 350,
        "savefig.dpi": 350,
        "axes.linewidth": 1.2,
    })
    
    palette = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16"]
    # Box plot color: unified dark neutral color for all box plots
    box_color = "#2C2C2C"  # Dark neutral gray
    
    # Geometry settings
    num_labels = len(labels)
    num_conditions = len(legend_labels)
    bar_width = 0.20
    inner_gap = 0.05
    group_gap = 1.0
    
    # Calculate x-positions
    x_positions = []
    for i in range(num_labels):
        base = i * (num_conditions * bar_width + (num_conditions - 1) * inner_gap + group_gap)
        for j in range(num_conditions):
            x_positions.append(base + j * (bar_width + inner_gap))
    x_positions = np.array(x_positions).reshape(num_labels, num_conditions)
    group_centers = x_positions.mean(axis=1)
    block_width = num_conditions * bar_width + (num_conditions - 1) * inner_gap
    
    def compute_ylim(values, margin_low=0.03, margin_high=0.03):
        vmin, vmax = float(values.min()), float(values.max())
        lo = max(0.0, vmin - margin_low)
        hi = min(1.0, vmax + margin_high)
        lo = np.floor(lo * 20) / 20.0
        hi = np.ceil(hi * 20) / 20.0
        if hi - lo < 0.20:
            hi = min(1.0, lo + 0.20)
        return lo, hi
    
    def build_boxplot_data(metric_name: str) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Build box plot data from all_metrics statistics.
        Returns dict mapping (label_idx, variant_idx) to box plot stats.
        """
        box_data = {}
        for var_idx, variant_key in enumerate(MODEL_ORDER):
            if variant_key not in all_metrics:
                continue
            variant_metrics = all_metrics[variant_key]
            for label_idx, label_info in enumerate(LABEL_CONSTRUCTS):
                label_key = label_info["key"]
                if label_key not in variant_metrics:
                    continue
                label_metrics = variant_metrics[label_key]
                if metric_name in label_metrics:
                    stat_values = label_metrics[metric_name]
                    if isinstance(stat_values, dict):
                        # Extract box plot statistics
                        box_stats = {}
                        for stat in ["min", "q1", "median", "q3", "max"]:
                            if stat in stat_values:
                                box_stats[stat] = stat_values[stat]
                        if len(box_stats) >= 3:  # Need at least min, median, max
                            box_data[(label_idx, var_idx)] = box_stats
        return box_data
    
    def draw_boxplot(ax, x, box_stats, color, width_factor=0.6):
        """
        Draw a box plot at position x using the box statistics.
        width_factor controls the width relative to bar_width.
        """
        if "min" not in box_stats or "max" not in box_stats:
            return
        
        min_val = box_stats["min"]
        max_val = box_stats["max"]
        median_val = box_stats.get("median", (min_val + max_val) / 2)
        q1_val = box_stats.get("q1", min_val)
        q3_val = box_stats.get("q3", max_val)
        
        box_width = bar_width * width_factor
        whisker_width = box_width * 0.3
        
        # Draw whiskers (min to max)
        ax.plot([x, x], [min_val, max_val], color=color, linewidth=0.8, zorder=4, clip_on=False, alpha=0.33)
        ax.plot(
            [x - whisker_width/2, x + whisker_width/2],
            [min_val, min_val],
            color=color,
            linewidth=0.8,
            zorder=4,
            clip_on=False,
            alpha=0.33
        )
        ax.plot(
            [x - whisker_width/2, x + whisker_width/2],
            [max_val, max_val],
            color=color,
            linewidth=0.8,
            zorder=4,
            clip_on=False,
            alpha=0.33
        )

        # Draw box (q1 to q3)
        if "q1" in box_stats and "q3" in box_stats:
            box_height = q3_val - q1_val
            box_bottom = q1_val
            # Draw box outline (no fill)
            rect = Rectangle(
                (x - box_width/2, box_bottom),
                box_width,
                box_height,
                linewidth=0.8,
                edgecolor=color,
                facecolor='none',
                zorder=4,
                clip_on=False,
                alpha=0.33
            )
            ax.add_patch(rect)

        # Draw median line
        ax.plot(
            [x - box_width/2, x + box_width/2],
            [median_val, median_val],
            color=color,
            linewidth=0.8,
            zorder=5,
            clip_on=False,
            alpha=0.33
        )
    
    # Determine figure size based on number of metrics
    num_metrics = len(selected_metrics)
    if num_metrics == 1:
        fig_width = 5.5
        fig_height = 3.6
    elif num_metrics == 2:
        fig_width = 11
        fig_height = 3.6
    elif num_metrics == 3:
        fig_width = 16.5
        fig_height = 3.6
    else:  # 4 metrics
        fig_width = 21
        fig_height = 3.6
    
    # Create figure
    if num_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=(fig_width, fig_height), sharey=False)
        axes = [axes]  # Make it iterable
    else:
        fig, axes = plt.subplots(1, num_metrics, figsize=(fig_width, fig_height), sharey=False)
    fig.patch.set_facecolor('#FAFAFB')
    custom_ylim = {"AUPRC": (0.40, 0.75)}
    
    for ax, (metric_name, metric_matrix) in zip(axes, filtered_metrics_data.items()):
        y_lo, y_hi = custom_ylim.get(metric_name, compute_ylim(metric_matrix))
        
        # Background bands
        for i in range(num_labels):
            ax.add_patch(Rectangle(
                (x_positions[i, 0] - bar_width / 2, y_lo),
                block_width,
                y_hi - y_lo,
                facecolor='#F4F6FA',
                edgecolor='none',
                zorder=0,
                alpha=1.0
            ))
        
        # Bars
        all_bars = []
        for c in range(num_conditions):
            bars = ax.bar(
                x_positions[:, c],
                metric_matrix[:, c],
                width=bar_width,
                color=palette[c],
                edgecolor='none',
                alpha=0.97,
                zorder=3,
                label=legend_labels[c]
            )
            all_bars.append(bars)
        
        # Box plots overlaid on bars
        box_data = build_boxplot_data(metric_name)
        for (label_idx, var_idx), box_stats in box_data.items():
            x_pos = x_positions[label_idx, var_idx]
            draw_boxplot(ax, x_pos, box_stats, box_color, width_factor=0.6)
        
        if num_metrics > 1:
            ax.set_title(metric_name, pad=6, fontsize=13, weight='bold', color='#1A1A1A')
        else:
            ax.set_ylabel(metric_name, fontsize=10, color='#2C2C2C')
        
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels, ha="center", fontsize=10, color='#2C2C2C')
        ax.yaxis.grid(True, linestyle=':', linewidth=0.8, alpha=0.7, color='#BFC7D5', zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#A9B2C3')
        ax.spines['bottom'].set_color('#A9B2C3')
    
    # Legend
    handles, _ = axes[0].get_legend_handles_labels()
    # Break legend into 2 lines if only 1 metric
    if num_metrics == 1:
        ncol = 2
        bbox_y = 1.08
    else:
        ncol = 4
        bbox_y = 1.03
    
    fig.legend(
        handles[:4],
        legend_labels,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, bbox_y),
        columnspacing=2.8,
        handlelength=1.8
    )
    
    plt.subplots_adjust(left=0.04, right=0.995, top=0.86, bottom=0.22, wspace=0.2)
    
    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{figure_name}.png")
    pdf_path = os.path.join(output_dir, f"{figure_name}.pdf")
    
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to: {png_path}")
    print(f"Saved figure to: {pdf_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ----------------------------- Main -----------------------------

def main():
    args = parse_args()
    
    # Determine input directory
    input_dir = args.input_dir if args.input_dir else args.output_dir
    
    all_metrics = {}
    
    if args.run_eval:
        print("=" * 60)
        print("Running Evaluation for SoSEX Ablation Study")
        print("=" * 60)
        
        # Process each model variant
        for variant_key in MODEL_ORDER:
            variant_info = MODEL_VARIANTS[variant_key]
            
            # Get config and checkpoint paths from args
            arg_prefix = variant_key.replace("_", "-")
            config_path = getattr(args, f"{variant_key}_config".replace("-", "_"))
            ckpt_path = getattr(args, f"{variant_key}_ckpt".replace("-", "_"))
            
            if config_path is None or ckpt_path is None:
                print(f"\nSkipping {variant_info['short_label']}: config or checkpoint not provided")
                continue
            
            # Resolve paths
            config_path = str(Path(config_path).resolve() if not Path(config_path).is_absolute() 
                             else config_path)
            ckpt_path = str(Path(ckpt_path).resolve() if not Path(ckpt_path).is_absolute() 
                           else ckpt_path)
            
            if not os.path.exists(config_path):
                print(f"\nSkipping {variant_info['short_label']}: config not found at {config_path}")
                continue
            
            if not os.path.exists(ckpt_path):
                print(f"\nSkipping {variant_info['short_label']}: checkpoint not found at {ckpt_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating: {variant_info['short_label']}")
            print(f"  Config: {config_path}")
            print(f"  Checkpoint: {ckpt_path}")
            print(f"{'='*60}")
            
            # Run evaluation
            try:
                eval_result = run_evaluation(config_path, ckpt_path, fold_idx=args.fold_idx)
                
                # Extract metrics
                extracted = extract_metrics_for_labels(
                    eval_result,
                    has_group_head=variant_info["grp_head"]
                )
                
                # Save to CSV
                save_metrics_to_csv(extracted, variant_key, args.output_dir)
                
                all_metrics[variant_key] = extracted
                
            except Exception as e:
                print(f"  Error evaluating {variant_info['short_label']}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    else:  # --use-saved
        print("=" * 60)
        print("Loading Pre-saved Results for SoSEX Ablation Study")
        print(f"Input directory: {input_dir}")
        print("=" * 60)
        
        for variant_key in MODEL_ORDER:
            variant_info = MODEL_VARIANTS[variant_key]
            
            try:
                metrics = load_metrics_from_csv(variant_key, input_dir)
                all_metrics[variant_key] = metrics
                print(f"  Loaded: {variant_info['short_label']}")
            except FileNotFoundError:
                print(f"  Skipping {variant_info['short_label']}: CSV file not found")
                continue
    
    if not all_metrics:
        print("\nError: No metrics available for visualization.")
        print("Please provide config/checkpoint pairs or ensure CSV files exist.")
        return
    
    # Build metrics data for visualization
    print("\n" + "=" * 60)
    print("Building Visualization")
    print("=" * 60)
    
    # Validate selected metrics
    valid_metrics = [m for m in args.metrics if m in METRICS_OF_INTEREST]
    if not valid_metrics:
        print(f"Error: No valid metrics selected. Available metrics: {list(METRICS_OF_INTEREST.keys())}")
        return
    
    print(f"Visualizing metrics: {', '.join(valid_metrics)}")
    
    metrics_data = build_metrics_data(all_metrics)
    
    # Create visualization
    visualize_ablation(
        metrics_data,
        all_metrics,
        valid_metrics,
        args.output_dir,
        figure_name=args.figure_name,
        show=not args.no_show
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
