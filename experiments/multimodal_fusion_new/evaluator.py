"""
Stress Test Evaluator

Evaluates trained fusion variants under systematic corruption sweeps.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils.metrics import compute_classification_metrics_from_logits
from engine.utils import modalities_to_branches
from experiments.multimodal_fusion_new.corruptions import (
    apply_uniform_corruption,
    apply_single_modality_corruption,
)
from experiments.multimodal_fusion_new.fusion_variants import variant_has_allocation_tracking


class StressTestEvaluator:
    """Evaluator for stress testing fusion variants."""
    
    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        variant_name: str = "glrx",
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.loader = loader
        self.device = device
        self.variant_name = variant_name
        self.logger = logger
        
        self.track_allocation = variant_has_allocation_tracking(variant_name)
        
        # Branch names (assumed: Videokinetic, Dialogue, Acoustic)
        self.branch_names = list(self.model.branches.keys())
    
    @torch.no_grad()
    def evaluate_clean(self) -> Dict[str, Any]:
        """Evaluate on clean data (no corruption)."""
        self.model.eval()
        
        all_preds = {
            split: {head: [] for head in ['Engagement', 'Valence']}
            for split in ['individual', 'group']
        }
        all_targets = {
            split: {head: [] for head in ['Engagement', 'Valence']}
            for split in ['individual', 'group']
        }
        
        allocation_stats = [] if self.track_allocation else None
        
        model_ref = getattr(self.model, "module", self.model)

        for batch_modalities, batch_labels in tqdm(
            self.loader, desc="Evaluating (clean)", leave=False
        ):
            # Move to device and group by branches (mirrors ReliabilityTrainer.validate)
            batch_modalities = {
                k: (v[0].to(self.device), v[1].to(self.device))
                for k, v in batch_modalities.items()
            }
            branch_data = modalities_to_branches(batch_modalities)
            batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}

            # Forward through frozen branches (no corruption)
            model_ref.branches.eval()
            z_branch: List[torch.Tensor] = []
            for bname, branch in model_ref.branches.items():
                key = bname.lower()
                if key not in branch_data:
                    continue
                mod_dict = branch_data[key]
                x_dict = {m: mod_dict[m][0] for m in branch.mods if m in mod_dict}
                m_dict = {m: mod_dict[m][1] for m in branch.mods if m in mod_dict}
                if not x_dict:
                    continue
                z_b = branch(x_dict, m_dict, epoch=None)
                z_branch.append(z_b)

            if not z_branch:
                continue

            # Multimodal fusion
            if len(z_branch) > 1:
                z = model_ref.mm_fusion(z_branch)
            else:
                z = z_branch[0]

            # Classification
            logits: Dict[str, Dict[str, torch.Tensor]] = {}
            if model_ref.individual_classifier:
                _, ind_logits = model_ref.individual_classifier(z, epoch=None)
                logits["individual"] = ind_logits
            if model_ref.group_classifier:
                _, grp_logits = model_ref.group_classifier(z, epoch=None)
                logits["group"] = grp_logits

            # Prepare labels (flatten individual labels to match training-time view)
            ind_labels = None
            if "individual" in batch_labels:
                B, P, L = batch_labels["individual"].shape
                ind_labels = batch_labels["individual"].view(B * P, L)
            grp_labels = batch_labels.get("group")

            # Collect predictions and targets
            for split in ["individual", "group"]:
                if split not in logits:
                    continue
                if split == "individual":
                    if ind_labels is None:
                        continue
                    head_labels = ind_labels
                else:
                    if grp_labels is None:
                        continue
                    head_labels = grp_labels

                for idx, (head, head_logits) in enumerate(logits[split].items()):
                    if head not in all_preds[split]:
                        # Skip unexpected heads to avoid key errors
                        continue
                    all_preds[split][head].append(head_logits.detach().cpu())
                    all_targets[split][head].append(
                        head_labels[:, idx].detach().cpu()
                    )

            # Allocation tracking
            if self.track_allocation:
                if hasattr(model_ref.mm_fusion, "get_aux"):
                    aux = model_ref.mm_fusion.get_aux()
                    if aux and "gate" in aux:
                        allocation_stats.append(aux["gate"].cpu().numpy())
        
        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_targets)
        
        # Allocation summary
        if allocation_stats:
            allocation_mean = np.concatenate(allocation_stats, axis=0).mean(axis=0)
            metrics['allocation'] = allocation_mean  # Shape: (M,)
        
        return metrics
    
    @torch.no_grad()
    def evaluate_corruption_sweep(
        self,
        corruption_type: str,
        param_values: List[float],
        modality_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Evaluate performance across a sweep of corruption parameters.
        
        Args:
            corruption_type: One of "dropout", "noise", "shuffle"
            param_values: List of corruption strengths to test
            modality_idx: If specified, only corrupt this modality; else corrupt all
            
        Returns:
            DataFrame with columns: [corruption_param, split, head, metric, value]
        """
        self.model.eval()
        
        results = []
        
        for param in tqdm(param_values, desc=f"Sweeping {corruption_type}", leave=False):
            all_preds = {
                split: {head: [] for head in ['Engagement', 'Valence']}
                for split in ['individual', 'group']
            }
            all_targets = {
                split: {head: [] for head in ['Engagement', 'Valence']}
                for split in ['individual', 'group']
            }
            
            allocation_stats = [] if self.track_allocation else None
            
            model_ref = getattr(self.model, "module", self.model)

            for batch_modalities, batch_labels in self.loader:
                # Move to device and group by branches
                batch_modalities = {
                    k: (v[0].to(self.device), v[1].to(self.device))
                    for k, v in batch_modalities.items()
                }
                branch_data = modalities_to_branches(batch_modalities)
                batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}
                
                # Forward through branches (clean first)
                model_ref.branches.eval()
                z_branch: List[torch.Tensor] = []
                for bname, branch in model_ref.branches.items():
                    key = bname.lower()
                    if key not in branch_data:
                        continue
                    mod_dict = branch_data[key]
                    x_dict = {m: mod_dict[m][0] for m in branch.mods if m in mod_dict}
                    m_dict = {m: mod_dict[m][1] for m in branch.mods if m in mod_dict}
                    if not x_dict:
                        continue
                    z_b = branch(x_dict, m_dict, epoch=None)
                    z_branch.append(z_b)
                
                if not z_branch:
                    continue
                
                # Apply corruption
                if modality_idx is not None:
                    z_branch_corrupted = apply_single_modality_corruption(
                        z_branch, modality_idx, corruption_type, param
                    )
                else:
                    z_branch_corrupted = apply_uniform_corruption(
                        z_branch, corruption_type, param
                    )
                
                # Forward through fusion
                if len(self.model.branches) > 1:
                    z = self.model.mm_fusion(z_branch_corrupted)
                else:
                    z = z_branch_corrupted[0]
                
                # Classification
                z_outs: Dict[str, Any] = {"backbone": z}
                logits: Dict[str, Dict[str, torch.Tensor]] = {}
                
                if model_ref.individual_classifier:
                    ind_features, ind_logits = model_ref.individual_classifier(z, epoch=None)
                    z_outs.update({'individual': ind_features})
                    logits.update({'individual': ind_logits})
                
                if model_ref.group_classifier:
                    grp_features, grp_logits = model_ref.group_classifier(z, epoch=None)
                    z_outs.update({'group': grp_features})
                    logits.update({'group': grp_logits})
                
                # Prepare labels (flatten individual labels to match training-time view)
                ind_labels = None
                if "individual" in batch_labels:
                    B, P, L = batch_labels["individual"].shape
                    ind_labels = batch_labels["individual"].view(B * P, L)
                grp_labels = batch_labels.get("group")

                # Collect predictions
                for split in ['individual', 'group']:
                    if split not in logits:
                        continue
                    if split == "individual":
                        if ind_labels is None:
                            continue
                        head_labels = ind_labels
                    else:
                        if grp_labels is None:
                            continue
                        head_labels = grp_labels

                    for idx, (head, head_logits) in enumerate(logits[split].items()):
                        if head not in all_preds[split]:
                            continue
                        all_preds[split][head].append(head_logits.detach().cpu())
                        all_targets[split][head].append(
                            head_labels[:, idx].detach().cpu()
                        )
                
                # Allocation tracking
                if self.track_allocation:
                    if hasattr(model_ref.mm_fusion, 'get_aux'):
                        aux = model_ref.mm_fusion.get_aux()
                        if aux and 'gate' in aux:
                            allocation_stats.append(aux['gate'].cpu().numpy())
            
            # Compute metrics for this param value
            metrics = self._compute_metrics(all_preds, all_targets)
            
            # Store results
            for split in ['individual', 'group']:
                for head in ['Engagement', 'Valence']:
                    prefix = f"{split}_{head}"
                    if f"{prefix}_f1" in metrics:
                        for metric_name in ['f1_macro', 'auroc', 'auprc']:
                            results.append({
                                'corruption_type': corruption_type,
                                'corruption_param': param,
                                'modality': self.branch_names[modality_idx] if modality_idx is not None else 'all',
                                'split': split,
                                'head': head,
                                'metric': metric_name,
                                'value': metrics.get(f"{prefix}_{metric_name.split('_')[-1]}", 0.0),
                            })
            
            # Store allocation if tracked
            if allocation_stats:
                allocation_mean = np.concatenate(allocation_stats, axis=0).mean(axis=0)
                for mod_idx, alloc_val in enumerate(allocation_mean):
                    results.append({
                        'corruption_type': corruption_type,
                        'corruption_param': param,
                        'modality': self.branch_names[modality_idx] if modality_idx is not None else 'all',
                        'split': 'allocation',
                        'head': self.branch_names[mod_idx],
                        'metric': 'allocation_weight',
                        'value': float(alloc_val),
                    })
        
        return pd.DataFrame(results)
    
    @torch.no_grad()
    def run_full_stress_test(
        self,
        corruption_configs: Optional[Dict[str, List[float]]] = None,
        per_modality: bool = True,
    ) -> pd.DataFrame:
        """
        Run comprehensive stress test with multiple corruption types.
        
        Args:
            corruption_configs: Dict mapping corruption_type -> list of param values
            per_modality: If True, also test per-modality corruption
            
        Returns:
            DataFrame with all results
        """
        if corruption_configs is None:
            corruption_configs = {
                'dropout': [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
                'noise': [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
                'shuffle': [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
            }
        
        all_results = []
        
        # Test uniform corruption (all modalities)
        if self.logger:
            self.logger.info("Running uniform corruption stress tests...")
        
        for corr_type, param_vals in corruption_configs.items():
            df = self.evaluate_corruption_sweep(corr_type, param_vals, modality_idx=None)
            all_results.append(df)
        
        # Test per-modality corruption
        if per_modality:
            if self.logger:
                self.logger.info("Running per-modality corruption stress tests...")
            
            for mod_idx, mod_name in enumerate(self.branch_names):
                if self.logger:
                    self.logger.info(f"  Testing {mod_name}...")
                
                for corr_type, param_vals in corruption_configs.items():
                    df = self.evaluate_corruption_sweep(
                        corr_type, param_vals, modality_idx=mod_idx
                    )
                    all_results.append(df)
        
        # Combine all results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _compute_metrics(
        self,
        all_preds: Dict[str, Dict[str, list]],
        all_targets: Dict[str, Dict[str, list]],
    ) -> Dict[str, float]:
        """Compute metrics for all heads."""
        metrics = {}
        
        for split in ['individual', 'group']:
            for head in ['Engagement', 'Valence']:
                if all_preds[split][head]:
                    preds = torch.cat(all_preds[split][head], dim=0)
                    targets = torch.cat(all_targets[split][head], dim=0)
                    
                    head_metrics = compute_classification_metrics_from_logits(preds, targets)
                    
                    prefix = f"{split}_{head}"
                    metrics[f"{prefix}_f1"] = head_metrics.get('f1_macro', 0.0)
                    metrics[f"{prefix}_auroc"] = head_metrics.get('auroc', 0.0)
                    metrics[f"{prefix}_auprc"] = head_metrics.get('auprc', 0.0)
        
        return metrics
    
    def _move_to_device(self, batch_data: dict) -> dict:
        """Recursively move batch data to device."""
        result = {}
        for key, value in batch_data.items():
            if isinstance(value, dict):
                result[key] = self._move_to_device(value)
            elif isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    result[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for v in value]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

