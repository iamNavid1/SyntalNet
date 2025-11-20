from __future__ import annotations
from typing import Dict, List, Optional, Any
import csv
import os
from pathlib import Path
import numpy as np
from collections import defaultdict


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
                    # Skip per-class metrics for now (can be added later)
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
        Compute mean and std across folds for each (variant, corruption, param, split, head, metric).
        
        Returns:
            Dict with keys "per_fold" and "aggregated"
        """
        # Group by (variant, corruption, param, split, head, metric)
        grouped = defaultdict(list)
        
        for row in self.results["all"]:
            key = (
                row["model_variant"],
                row["corruption_type"],
                row["corruption_param"],
                row["split"],
                row["head"],
                row["metric"],
            )
            grouped[key].append(row["value"])
        
        # Compute statistics
        aggregated = []
        for key, values in grouped.items():
            variant, corr_type, corr_param, split, head, metric = key
            values_arr = np.array(values)
            
            aggregated.append({
                "model_variant": variant,
                "corruption_type": corr_type,
                "corruption_param": corr_param,
                "split": split,
                "head": head,
                "metric": metric,
                "mean": float(np.mean(values_arr)),
                "std": float(np.std(values_arr)),
                "min": float(np.min(values_arr)),
                "max": float(np.max(values_arr)),
                "n_folds": len(values),
            })
        
        return {
            "per_fold": self.results["all"],
            "aggregated": aggregated,
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

