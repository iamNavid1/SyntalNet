from __future__ import annotations
from typing import Dict, List, Optional, Any
import csv
import os
from pathlib import Path
import numpy as np
from collections import defaultdict


SELECTED_METRICS = {
    "accuracy",
    "f1_macro",
    "auroc_macro",
    "auprc_macro",
}


class MetricsCollector:
    """
    Collects metrics across folds and corruption scenarios.
    """
    def __init__(self):
        self.results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def add_result(
        self,
        model_variant: str,
        corruption_type: str,
        corruption_param: float | int,
        fold_idx: int,
        metrics: Dict[str, Dict[str, Dict[str, float]]],
    ):
        """
        Add a result from a single evaluation run.
        
        Args:
            model_variant: Name of model variant (e.g., "bscx", "proj_only")
            corruption_type: Type of corruption (e.g., "stream_dropout")
            corruption_param: Parameter value for corruption
            fold_idx: Fold index (0-4)
            metrics: Nested dict {split: {head: {metric: value}}}
        """
        # Flatten metrics structure
        for split in metrics:
            for head_name in metrics[split]:
                for metric_name, metric_value in metrics[split][head_name].items():
                    if metric_name not in SELECTED_METRICS:
                        continue
                    
                    # Skip per-class metrics (lists/arrays)
                    if isinstance(metric_value, (list, np.ndarray)):
                        continue
                    
                    self.results["all"].append({
                        "model_variant": model_variant,
                        "corruption_type": corruption_type,
                        "corruption_param": corruption_param,
                        "fold": fold_idx,
                        "split": split,
                        "head": head_name,
                        "metric": metric_name,
                        "value": float(metric_value),
                    })
    
    def compute_statistics(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compute comprehensive statistics across folds.
        
        Three levels of aggregation:
        1. Per-construct (split-head): aggregate over folds for each construct separately
        2. All-constructs by split: aggregate over all constructs within each split (individual or group)
        3. All-constructs combined: aggregate over ALL constructs (both individual and group) × folds
        
        Returns:
            Dict with keys "per_fold", "per_construct", and "all_constructs"
        """
        # Group by (variant, corruption, param, split, head, metric) for per-construct stats
        per_construct_grouped = defaultdict(list)
        
        # Group by (variant, corruption, param, split, metric) for all-constructs by split
        # This aggregates all constructs within each split separately (individual: 2, group: 3)
        all_constructs_by_split_grouped = defaultdict(list)
        
        # Group by (variant, corruption, param, metric) for all-constructs combined
        # This aggregates across BOTH individual and group splits (all 5 constructs)
        all_constructs_combined_grouped = defaultdict(list)
        
        for row in self.results["all"]:
            # Per-construct key (includes split and head)
            key_per_construct = (
                row["model_variant"],
                row["corruption_type"],
                row["corruption_param"],
                row["split"],
                row["head"],
                row["metric"],
            )
            per_construct_grouped[key_per_construct].append(row["value"])
            
            # All-constructs by split key (excludes head, but includes split)
            # This aggregates all constructs within each split (individual: 2, group: 3)
            key_all_constructs_by_split = (
                row["model_variant"],
                row["corruption_type"],
                row["corruption_param"],
                row["split"],
                row["metric"],
            )
            all_constructs_by_split_grouped[key_all_constructs_by_split].append(row["value"])
            
            # All-constructs combined key (excludes split and head, aggregates across ALL constructs)
            # This combines both individual (2 constructs) and group (3 constructs) = 5 total
            key_all_constructs_combined = (
                row["model_variant"],
                row["corruption_type"],
                row["corruption_param"],
                row["metric"],
            )
            all_constructs_combined_grouped[key_all_constructs_combined].append(row["value"])
        
        # Compute per-construct statistics (aggregate over folds)
        per_construct_aggregated = []
        for key, values in per_construct_grouped.items():
            variant, corr_type, corr_param, split, head, metric = key
            values_arr = np.array(values)
            
            per_construct_aggregated.append({
                "model_variant": variant,
                "corruption_type": corr_type,
                "corruption_param": corr_param,
                "split": split,
                "head": head,
                "metric": metric,
                "mean": float(np.mean(values_arr)),
                "std": float(np.std(values_arr, ddof=1)),  # Sample std
                "min": float(np.min(values_arr)),
                "max": float(np.max(values_arr)),
                "median": float(np.median(values_arr)),
                "q1": float(np.percentile(values_arr, 25)),
                "q3": float(np.percentile(values_arr, 75)),
                "n_folds": len(values),
            })
        
        # Compute all-constructs statistics (two types)
        all_constructs_aggregated = []
        
        # Type a) All-constructs by split (aggregates all constructs within each split)
        # Individual: 2 constructs × folds, Group: 3 constructs × folds
        for key, values in all_constructs_by_split_grouped.items():
            variant, corr_type, corr_param, split, metric = key
            values_arr = np.array(values)
            
            all_constructs_aggregated.append({
                "model_variant": variant,
                "corruption_type": corr_type,
                "corruption_param": corr_param,
                "split": split,  # "individual" or "group"
                "head": "all_constructs",  # Special marker
                "metric": metric,
                "mean": float(np.mean(values_arr)),
                "std": float(np.std(values_arr, ddof=1)),  # Sample std
                "min": float(np.min(values_arr)),
                "max": float(np.max(values_arr)),
                "median": float(np.median(values_arr)),
                "q1": float(np.percentile(values_arr, 25)),
                "q3": float(np.percentile(values_arr, 75)),
                "n_folds": len(values),  # Total number of values (constructs in split × folds)
            })
        
        # Type b) All-constructs combined (aggregates across both splits)
        # This aggregates across both individual (2) and group (3) splits = 5 constructs total
        for key, values in all_constructs_combined_grouped.items():
            variant, corr_type, corr_param, metric = key
            values_arr = np.array(values)
            
            all_constructs_aggregated.append({
                "model_variant": variant,
                "corruption_type": corr_type,
                "corruption_param": corr_param,
                "split": "all",  # Aggregates across both individual and group splits
                "head": "all_constructs",  # Special marker
                "metric": metric,
                "mean": float(np.mean(values_arr)),
                "std": float(np.std(values_arr, ddof=1)),  # Sample std
                "min": float(np.min(values_arr)),
                "max": float(np.max(values_arr)),
                "median": float(np.median(values_arr)),
                "q1": float(np.percentile(values_arr, 25)),
                "q3": float(np.percentile(values_arr, 75)),
                "n_folds": len(values),  # Total number of values (5 constructs × folds)
            })
        
        return {
            "per_fold": self.results["all"],
            "per_construct": per_construct_aggregated,
            "all_constructs": all_constructs_aggregated,
        }
    
    def export_csv(
        self,
        output_dir: str,
        filename_prefix: str = "stress_test_results",
    ):
        """
        Export results to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
            filename_prefix: Prefix for output files
        
        Returns:
            Tuple of (per_fold_path, per_construct_path, all_constructs_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stats = self.compute_statistics()
        
        # Export per-fold results
        per_fold_path = os.path.join(output_dir, f"{filename_prefix}_per_fold.csv")
        self._write_csv(per_fold_path, stats["per_fold"])
        
        # Export per-construct aggregated results
        per_construct_path = os.path.join(output_dir, f"{filename_prefix}_per_construct.csv")
        self._write_csv(per_construct_path, stats["per_construct"])
        
        # Export all-constructs aggregated results
        all_constructs_path = os.path.join(output_dir, f"{filename_prefix}_all_constructs.csv")
        self._write_csv(all_constructs_path, stats["all_constructs"])
        
        return per_fold_path, per_construct_path, all_constructs_path
    
    def _write_csv(self, filepath: str, rows: List[Dict[str, Any]]):
        """Write rows to CSV file."""
        if not rows:
            return
        
        fieldnames = list(rows[0].keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def get_baseline_key(self) -> tuple:
        """Return the key that represents baseline (no corruption)."""
        return ("baseline", None, 0.0, None, None, None)
    
    def is_baseline(
        self,
        corruption_type: str,
        corruption_param: float | int,
    ) -> bool:
        """
        Check if a corruption configuration represents baseline (no corruption).
        """
        baseline_conditions = {
            "stream_dropout": corruption_param == 0.0,
            "channel_dropout": corruption_param == 0.0,
            "temporal_band": corruption_param == 0.0,
            "jitter": corruption_param == 0,
            "energy": corruption_param == 1.0,
        }
        return baseline_conditions.get(corruption_type, False)

