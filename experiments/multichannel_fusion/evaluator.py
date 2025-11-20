from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from engine.utils import modalities_to_branches, BuildAutocastKWargs
from experiments.multichannel_fusion.model_wrapper import CorruptedSyntalNet, create_corruption_fn
from utils.metrics import build_classification_metrics, compute_metrics

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class StressTestEvaluator:
    """
    Evaluator that runs models with corruption injection and collects metrics.
    """
    def __init__(
        self,
        device: torch.device,
        cfg: dict,
        collect_individual: bool = True,
        collect_group: bool = True,
    ):
        self.device = device
        self.cfg = cfg
        self.collect_individual = collect_individual
        self.collect_group = collect_group
        self.autocast_kwargs = BuildAutocastKWargs(cfg, device)
    
    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        corruption_type: Optional[str] = None,
        corruption_param: Optional[float | int] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate model on dataloader with optional corruption.
        
        Args:
            model: Base SyntalNet model
            dataloader: DataLoader for evaluation
            corruption_type: Type of corruption to apply (None = no corruption)
            corruption_param: Parameter for corruption
        
        Returns:
            Nested dict: {split: {head_name: {metric_name: value}}}
        """
        # Create corruption function
        corruption_fn = create_corruption_fn(corruption_type, corruption_param)
        
        # Wrap model with corruption
        if corruption_fn is not None:
            eval_model = CorruptedSyntalNet(model, corruption_fn)
        else:
            eval_model = model
        
        eval_model.eval()
        eval_model = eval_model.to(self.device)
        
        # Build metric sets
        metric_sets = self._build_metric_sets(eval_model)
        if not metric_sets:
            return {}
        
        # Run evaluation
        for batch_data, batch_labels in dataloader:
            batch_data = {
                k: (v[0].to(self.device), v[1].to(self.device))
                for k, v in batch_data.items()
            }
            batch_data = modalities_to_branches(batch_data)
            batch_labels = {
                k: v.to(self.device) for k, v in batch_labels.items()
            }
            
            with autocast(**self.autocast_kwargs):
                _, logits = eval_model(batch_data)
            
            # Update metrics
            for split in ("individual", "group"):
                if split in metric_sets and split in logits and split in batch_labels:
                    self._update_metrics(
                        metric_sets[split],
                        logits[split],
                        batch_labels[split]
                    )
        
        # Compute results
        results = self._compute_results(metric_sets)
        return results
    
    def _build_metric_sets(self, model: torch.nn.Module):
        """Build metric sets for all classification heads."""
        metric_sets: Dict[str, Dict[str, Dict[str, torch.nn.Module]]] = {}
        
        # Get base model (unwrap if needed)
        base_model = getattr(model, "base_model", model)
        
        if self.collect_individual and getattr(base_model, "individual_classifier", None) is not None:
            metric_sets["individual"] = {}
            for name, clf in base_model.individual_classifier.classifiers.items():
                num_classes = clf.K
                metrics = build_classification_metrics(num_classes)
                for m in metrics.values():
                    m.to(torch.device('cpu'))
                metric_sets["individual"][name] = metrics
        
        if self.collect_group and getattr(base_model, "group_classifier", None) is not None:
            metric_sets["group"] = {}
            for name, clf in base_model.group_classifier.classifiers.items():
                num_classes = clf.K
                metrics = build_classification_metrics(num_classes)
                for m in metrics.values():
                    m.to(torch.device('cpu'))
                metric_sets["group"][name] = metrics
        
        return metric_sets
    
    def _update_metrics(
        self,
        metric_sets_split: Dict[str, Dict[str, torch.nn.Module]],
        logits_split: Dict[str, torch.Tensor],
        labels_tensor: torch.Tensor,
    ):
        """Update metrics for a split (individual or group)."""
        metrics_device = torch.device('cpu')
        
        # Handle label shape
        if labels_tensor.dim() == 3:  # individual: (B, P, L)
            B, P, L = labels_tensor.shape
            lbl = labels_tensor.view(B * P, L)
        else:  # group: (B, L)
            lbl = labels_tensor
        
        lbl = lbl.to(metrics_device)
        prob_metrics = {"auroc_macro", "auprc_macro", "auprc_per_class", "ece"}
        
        for idx, (name, lg) in enumerate(logits_split.items()):
            lg_cpu = lg.detach().to(metrics_device)
            y = lbl[:, idx].detach().to(metrics_device)
            probs = torch.softmax(lg_cpu, dim=-1)
            
            metrics = metric_sets_split[name]
            for mname, m in metrics.items():
                if mname in prob_metrics:
                    m.update(probs, y)
                else:
                    m.update(lg_cpu, y)
    
    def _compute_results(self, metric_sets: Dict[str, Dict[str, Dict[str, torch.nn.Module]]]):
        """Compute final metric results."""
        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        for split, heads in metric_sets.items():
            results[split] = {}
            for head_name, metrics in heads.items():
                results[split][head_name] = compute_metrics(metrics)
        
        return results

