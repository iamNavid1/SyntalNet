from __future__ import annotations

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import defaultdict


class OODAdaptationMetricsCollector:
    """
    Collects and organizes metrics from OOD adaptation experiments.
    """
    
    def __init__(self):
        # Storage: fold -> proportion -> epoch -> metrics
        self.results: Dict[int, Dict[float, Dict[int, Dict]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        
        # Metadata
        self.metadata = {
            "experiment_type": "ood_adaptation",
            "base_model": None,
            "n_folds": None,
        }
    
    def add_result(
        self,
        fold_idx: int,
        proportion: float,
        epoch: int,
        metrics: Dict,
        train_loss: Optional[float] = None,
    ):
        """
        Add results for a specific fold, proportion, and epoch.
        
        Args:
            fold_idx: LOGO fold index (held-out group ID)
            proportion: Proportion of left-out data used for fine-tuning
            epoch: Fine-tuning epoch
            metrics: Validation metrics dictionary
            train_loss: Training loss (optional)
        """
        result = {
            "metrics": metrics,
            "train_loss": train_loss,
        }
        self.results[fold_idx][proportion][epoch] = result
    
    def export_csv(
        self,
        output_dir: str,
        filename_prefix: str = "ood_adaptation"
    ) -> tuple[str, str]:
        """
        Export results to CSV files.
        
        Creates two CSV files:
        1. Per-epoch results for all folds and proportions (includes train_loss and all validation metrics)
        2. Aggregated results across folds for each proportion and epoch
           - Statistics: mean, std, min, max, median, q1, q3
           - Includes both train_loss and validation metrics
        
        Args:
            output_dir: Directory to save CSV files
            filename_prefix: Prefix for output filenames
        
        Returns:
            Tuple of (per_epoch_csv_path, aggregated_csv_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Per-epoch results ---
        rows = []
        for fold_idx in sorted(self.results.keys()):
            for proportion in sorted(self.results[fold_idx].keys()):
                for epoch in sorted(self.results[fold_idx][proportion].keys()):
                    result = self.results[fold_idx][proportion][epoch]
                    metrics = result["metrics"]
                    
                    row = {
                        "fold": fold_idx,
                        "proportion": proportion,
                        "epoch": epoch,
                        "train_loss": result.get("train_loss"),
                    }
                    
                    # Flatten metrics
                    flat_metrics = self._flatten_metrics(metrics)
                    row.update(flat_metrics)
                    
                    rows.append(row)
        
        per_epoch_df = pd.DataFrame(rows)
        per_epoch_path = os.path.join(output_dir, f"{filename_prefix}_per_epoch.csv")
        per_epoch_df.to_csv(per_epoch_path, index=False)
        
        # --- Aggregated results ---
        agg_rows = []
        
        # Get all unique (proportion, epoch) pairs
        proportion_epoch_pairs = set()
        for fold_data in self.results.values():
            for proportion in fold_data.keys():
                for epoch in fold_data[proportion].keys():
                    proportion_epoch_pairs.add((proportion, epoch))
        
        for proportion, epoch in sorted(proportion_epoch_pairs):
            # Collect metrics from all folds for this (proportion, epoch)
            fold_metrics = []
            fold_train_losses = []
            for fold_idx in sorted(self.results.keys()):
                if proportion in self.results[fold_idx] and epoch in self.results[fold_idx][proportion]:
                    result = self.results[fold_idx][proportion][epoch]
                    metrics = result["metrics"]
                    flat_metrics = self._flatten_metrics(metrics)
                    fold_metrics.append(flat_metrics)
                    
                    # Collect train_loss separately
                    train_loss = result.get("train_loss")
                    if train_loss is not None:
                        fold_train_losses.append(train_loss)
            
            if not fold_metrics:
                continue
            
            # Compute statistics across folds
            agg_row = {
                "proportion": proportion,
                "epoch": epoch,
                "n_folds": len(fold_metrics),
            }
            
            # Aggregate train_loss
            if fold_train_losses:
                import numpy as np
                agg_row["train_loss_mean"] = np.mean(fold_train_losses)
                agg_row["train_loss_std"] = np.std(fold_train_losses)
                agg_row["train_loss_min"] = np.min(fold_train_losses)
                agg_row["train_loss_max"] = np.max(fold_train_losses)
                agg_row["train_loss_median"] = np.median(fold_train_losses)
                agg_row["train_loss_q1"] = np.percentile(fold_train_losses, 25)
                agg_row["train_loss_q3"] = np.percentile(fold_train_losses, 75)
            
            # Get all metric keys
            all_keys = set()
            for fm in fold_metrics:
                all_keys.update(fm.keys())
            
            for key in sorted(all_keys):
                values = [fm.get(key) for fm in fold_metrics if key in fm]
                if values and all(isinstance(v, (int, float)) for v in values):
                    import numpy as np
                    agg_row[f"{key}_mean"] = np.mean(values)
                    agg_row[f"{key}_std"] = np.std(values)
                    agg_row[f"{key}_min"] = np.min(values)
                    agg_row[f"{key}_max"] = np.max(values)
                    agg_row[f"{key}_median"] = np.median(values)
                    agg_row[f"{key}_q1"] = np.percentile(values, 25)
                    agg_row[f"{key}_q3"] = np.percentile(values, 75)
            
            agg_rows.append(agg_row)
        
        agg_df = pd.DataFrame(agg_rows)
        agg_path = os.path.join(output_dir, f"{filename_prefix}_aggregated.csv")
        agg_df.to_csv(agg_path, index=False)
        
        return per_epoch_path, agg_path
    
    def export_json(self, output_path: str):
        """Export raw results to JSON."""
        # Convert defaultdicts to regular dicts for JSON serialization
        def convert_to_dict(obj):
            if isinstance(obj, defaultdict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            return obj
        
        output_data = {
            "metadata": self.metadata,
            "results": convert_to_dict(self.results),
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
    
    def _flatten_metrics(self, metrics: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested metrics dictionary."""
        flat = {}
        for key, value in metrics.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten
                flat.update(self._flatten_metrics(value, prefix=f"{new_key}_"))
            elif isinstance(value, (list, tuple)):
                # Skip lists (like confusion matrices)
                continue
            elif isinstance(value, (int, float, str, bool)):
                flat[new_key] = value
            else:
                # Try to convert to float
                try:
                    flat[new_key] = float(value)
                except (TypeError, ValueError):
                    continue
        
        return flat
    
    def get_best_epoch_per_fold(
        self,
        metric_key: str = "loss",
        minimize: bool = True
    ) -> Dict[int, Dict[float, int]]:
        """
        Find the best epoch for each fold and proportion based on a metric.
        
        Args:
            metric_key: Metric to optimize (e.g., "loss", "individual_Engagement_accuracy")
            minimize: Whether to minimize (True) or maximize (False) the metric
        
        Returns:
            Dict mapping fold_idx -> proportion -> best_epoch
        """
        best_epochs = {}
        
        for fold_idx in self.results.keys():
            best_epochs[fold_idx] = {}
            for proportion in self.results[fold_idx].keys():
                best_epoch = None
                best_value = float('inf') if minimize else float('-inf')
                
                for epoch, result in self.results[fold_idx][proportion].items():
                    metrics = result["metrics"]
                    flat_metrics = self._flatten_metrics(metrics)
                    
                    if metric_key in flat_metrics:
                        value = flat_metrics[metric_key]
                        if minimize:
                            if value < best_value:
                                best_value = value
                                best_epoch = epoch
                        else:
                            if value > best_value:
                                best_value = value
                                best_epoch = epoch
                
                if best_epoch is not None:
                    best_epochs[fold_idx][proportion] = best_epoch
        
        return best_epochs

