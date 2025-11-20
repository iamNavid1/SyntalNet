from __future__ import annotations
from typing import Dict, List, Optional, Any
import csv
import os
import numpy as np
from collections import defaultdict


class SOSEXMetricsCollector:
    """
    Collects metrics across folds and model variants for SOSEX experiments.
    Only collects: AUPRC macro, AUROC macro, balanced accuracy, F1 macro.
    """
    
    # Metrics to extract
    TARGET_METRICS = ["auprc_macro", "auroc_macro", "balanced_accuracy", "f1_macro"]
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def add_result(
        self,
        variant_name: str,
        fold_idx: Optional[int],
        metrics: Dict[str, Dict[str, Dict[str, float]]],
    ):
        """
        Add a result from a single evaluation run.
        
        Args:
            variant_name: Name of model variant (e.g., "w-mp-fus_wo-grp-cls")
            fold_idx: Fold index (None for single fold, int for kfold/logo)
            metrics: Nested dict {split: {head: {metric: value}}}
        """
        # Extract only target metrics for all heads
        for split in metrics:
            for head_name in metrics[split]:
                for metric_name, metric_value in metrics[split][head_name].items():
                    # Only collect target metrics
                    if metric_name not in self.TARGET_METRICS:
                        continue
                    
                    # Skip per-class metrics (arrays/lists)
                    if isinstance(metric_value, (list, np.ndarray)):
                        continue
                    
                    self.results.append({
                        "variant": variant_name,
                        "fold": fold_idx if fold_idx is not None else "single",
                        "split": split,
                        "head": head_name,
                        "metric": metric_name,
                        "value": float(metric_value),
                    })
    
    def compute_statistics(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compute statistics across folds for each (variant, split, head, metric).
        
        Returns:
            Dict with keys "per_fold" and "aggregated"
        """
        # Group by (variant, split, head, metric)
        grouped = defaultdict(list)
        
        for row in self.results:
            key = (
                row["variant"],
                row["split"],
                row["head"],
                row["metric"],
            )
            grouped[key].append(row["value"])
        
        # Compute statistics
        aggregated = []
        for key, values in grouped.items():
            variant, split, head, metric = key
            values_arr = np.array(values)
            
            aggregated.append({
                "variant": variant,
                "split": split,
                "head": head,
                "metric": metric,
                "mean": float(np.mean(values_arr)),
                "std": float(np.std(values_arr)),
                "min": float(np.min(values_arr)),
                "max": float(np.max(values_arr)),
                "median": float(np.median(values_arr)),
                "q1": float(np.percentile(values_arr, 25)),
                "q3": float(np.percentile(values_arr, 75)),
                "n_folds": len(values),
            })
        
        return {
            "per_fold": self.results,
            "aggregated": aggregated,
        }
    
    def export_csv(
        self,
        output_dir: str,
        filename_prefix: str = "sosex_experiments_results",
    ):
        """
        Export results to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
            filename_prefix: Prefix for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stats = self.compute_statistics()
        
        # Export per-fold results
        per_fold_path = os.path.join(output_dir, f"{filename_prefix}_per_fold.csv")
        self._write_csv(per_fold_path, stats["per_fold"])
        
        # Export aggregated results
        aggregated_path = os.path.join(output_dir, f"{filename_prefix}_aggregated.csv")
        self._write_csv(aggregated_path, stats["aggregated"])
        
        return per_fold_path, aggregated_path
    
    def _write_csv(self, filepath: str, rows: List[Dict[str, Any]]):
        """Write rows to CSV file."""
        if not rows:
            return
        
        fieldnames = list(rows[0].keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

