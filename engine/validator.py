from __future__ import annotations

from typing import Dict, Union
import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data.distributed import DistributedSampler

from utils.metrics import build_classification_metrics, compute_metrics


class Validator:
    def __init__(self, device: torch.device, autocast_kwargs: dict):
        self.device = device
        self.autocast_kwargs = autocast_kwargs

    # ----------------------------- Public API -----------------------------

    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module,
        dataloader,
    ):
        model.eval()

        metric_sets = self._build_metric_sets(model)
        if not metric_sets:
            return {}

        ddp_eval = self._is_ddp_eval(dataloader)

        if ddp_eval:
            head_specs = self._get_head_specs(model)  # {split:{head:K}}
            buffers = {split: {name: {"logits": [], "targets": []} for name in heads}
                       for split, heads in head_specs.items()}
        else:
            head_specs, buffers = None, None

        for batch_data, batch_labels in dataloader:
            batch_data = {k: (v[0].to(self.device), v[1].to(self.device)) for k, v in batch_data.items()}
            batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}

            with autocast(**self.autocast_kwargs):
                _, logits = model(batch_data)

            if ddp_eval:
                for k in ("individual", "group"):
                    if k in buffers and k in logits and k in batch_labels:
                        self._update_buffers(buffers[k], logits[k], batch_labels[k])
            else:
                for k in ("individual", "group"):
                    if k in metric_sets and k in logits and k in batch_labels:
                        self._update_metric(metric_sets[k], logits[k], batch_labels[k])

        # non-DDP: compute results from local metric modules
        if not ddp_eval:
            return self._metrics_results(metric_sets)

        # DDP: gather per-head tensors to rank 0, compute once, broadcast
        gathered = self._gather_per_head(buffers, head_specs)
        if dist.get_rank() == 0:
            results = self._buffers_results(head_specs, gathered)
        else:
            results = None

        obj_list = [results]
        dist.broadcast_object_list(obj_list, src=0)
        results = obj_list[0]
        return results

    __call__ = run 

    # --------------------------- Helpers ---------------------------

    def _is_ddp_eval(self, dataloader) -> bool:
        return (
            dist.is_available()
            and dist.is_initialized()
            and hasattr(dataloader, "sampler")
            and isinstance(dataloader.sampler, DistributedSampler)
        )

    def _get_head_specs(self, model):
        specs: Dict[str, Dict[str, int]] = {}
        if getattr(model, "individual_classifier", None) is not None:
            specs["individual"] = {name: clf.K for name, clf in model.individual_classifier.classifiers.items()}
        if getattr(model, "group_classifier", None) is not None:
            specs.setdefault("group", {})
            specs["group"] = {name: clf.K for name, clf in model.group_classifier.classifiers.items()}
        return specs

    def _build_metric_sets(self, model):
        metric_sets: Dict[str, Dict[str, Dict[str, torch.nn.Module]]] = {}

        if getattr(model, "individual_classifier", None) is not None:
            metric_sets["individual"] = {}
            for name, clf in model.individual_classifier.classifiers.items():
                metrics = build_classification_metrics(clf.K)
                for m in metrics.values():
                    m.to(self.device)
                metric_sets["individual"][name] = metrics

        if getattr(model, "group_classifier", None) is not None:
            metric_sets["group"] = {}
            for name, clf in model.group_classifier.classifiers.items():
                metrics = build_classification_metrics(clf.K)
                for m in metrics.values():
                    m.to(self.device)
                metric_sets["group"][name] = metrics

        return metric_sets

    # --------------------------- Update paths ---------------------------

    def _update_buffers(
            self,
            buffers_sets_split,
            logits_split: Dict[str, torch.Tensor],
            labels_tensor: torch.Tensor,
    ):
        if labels_tensor.dim() == 3:  # individual shape (B, P, L)
            B, P, L = labels_tensor.shape
            lbl = labels_tensor.view(B * P, L)
        else:  # group shape (B, L)
            lbl = labels_tensor
        for idx, (name, lg) in enumerate(logits_split.items()):
            y = lbl[:, idx]
            buffers = buffers_sets_split[name]
            buffers["logits"].append(lg.detach().cpu())
            buffers["targets"].append(y.detach().cpu())

    def _update_metric(
            self,
            metric_sets_split: Dict[str, Dict[str, torch.nn.Module]],
            logits_split: Dict[str, torch.Tensor],
            labels_tensor: torch.Tensor,
    ):
        if labels_tensor.dim() == 3:  # individual shape (B, P, L)
            B, P, L = labels_tensor.shape
            lbl = labels_tensor.view(B * P, L)
        else:  # group shape (B, L)
            lbl = labels_tensor
        for idx, (name, lg) in enumerate(logits_split.items()):
            y = lbl[:, idx]
            probs = torch.softmax(lg, dim=-1)
            metrics = metric_sets_split[name]
            metrics["accuracy"].update(lg, y)
            metrics["f1_macro"].update(lg, y)
            metrics["f1"].update(lg, y)
            metrics["precision_macro"].update(lg, y)
            metrics["auroc_ovr"].update(probs, y)
            metrics["auprc_macro"].update(probs, y)
            metrics["ece"].update(probs, y)
            if "confusion_matrix" in metrics:
                metrics["confusion_matrix"].update(lg, y)

    # --------------------------- Results computation ---------------------------

    def _metrics_results(self, metric_sets):
        results: Dict[str, Dict[str, Dict[str, Union[float, np.ndarray]]]] = {}
        for group_name, heads in metric_sets.items():
            results[group_name] = {}
            for head_name, metrics in heads.items():
                results[group_name][head_name] = compute_metrics(metrics)
        return results

    def _buffers_results(
        self,
        head_specs: Dict[str, Dict[str, int]],
        gathered: Dict[str, Dict[str, tuple]],
    ):
        results: Dict[str, Dict[str, Dict[str, Union[float, np.ndarray]]]] = {}
        for split, heads in head_specs.items():
            results[split] = {}
            for name, K in heads.items():
                lg, y = gathered[split][name]
                lg = lg.float()
                metrics = build_classification_metrics(K)  # on cpu
                if lg.numel() > 0:
                    probs = torch.softmax(lg, dim=-1)
                    metrics["accuracy"].update(lg, y)
                    metrics["f1_macro"].update(lg, y)
                    metrics["f1"].update(lg, y)
                    metrics["precision_macro"].update(lg, y)
                    metrics["auroc_ovr"].update(probs, y)
                    metrics["auprc_macro"].update(probs, y)
                    metrics["ece"].update(probs, y)
                    if "confusion_matrix" in metrics:
                        metrics["confusion_matrix"].update(lg, y)
                results[split][name] = compute_metrics(metrics)
        return results

    # --------------------------- DDP gather ---------------------------

    def _gather_per_head(
        self,
        buffers: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        head_specs: Dict[str, Dict[str, int]],
    ):
        for split in buffers:
            for name in buffers[split]:
                if buffers[split][name]["logits"]:
                    buffers[split][name]["logits"] = torch.cat(buffers[split][name]["logits"], dim=0)
                    buffers[split][name]["targets"] = torch.cat(buffers[split][name]["targets"], dim=0)
                else:
                    K = head_specs[split][name]
                    buffers[split][name]["logits"] = torch.empty(0, K)
                    buffers[split][name]["targets"] = torch.empty(0, dtype=torch.long)

        world_size = dist.get_world_size()
        gathered: Dict[str, Dict[str, tuple]] = {}
        rank = dist.get_rank()

        for split in head_specs:
            gathered[split] = {}
            for name, K in head_specs[split].items():
                lg_list = [None for _ in range(world_size)]
                y_list = [None for _ in range(world_size)]
                lg_list = dist.all_gather_object(lg_list, buffers[split][name]["logits"])
                y_list = dist.all_gather_object(y_list, buffers[split][name]["targets"])
                if rank == 0:
                    lg_all = torch.cat([x for x in lg_list if x is not None and x.numel() > 0], dim=0) if any(
                        (x is not None and x.numel() > 0) for x in lg_list
                    ) else torch.empty(0, K)
                    y_all = torch.cat([x for x in y_list if x is not None and x.numel() > 0], dim=0) if any(
                        (x is not None and x.numel() > 0) for x in y_list
                    ) else torch.empty(0, dtype=torch.long)
                    gathered[split][name] = (lg_all, y_all)

        return gathered if rank == 0 else None
