from __future__ import annotations

from typing import Dict, Any, Optional

import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassMatthewsCorrCoef,
    MulticlassSpecificity,
)


def build_classification_metrics(num_classes: int) -> Dict[str, torch.nn.Module]:
    """Create a dictionary of classification metrics for ``num_classes`` classes."""

    return {
        "accuracy": MulticlassAccuracy(num_classes=num_classes),
        "balanced_accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro"),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=num_classes),

        "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "precision_micro": MulticlassPrecision(num_classes=num_classes, average="micro"),
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall_micro": MulticlassRecall(num_classes=num_classes, average="micro"),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),

        "f1_per_class": MulticlassF1Score(num_classes=num_classes, average=None),
        "precision_per_class": MulticlassPrecision(num_classes=num_classes, average=None),
        "recall_per_class": MulticlassRecall(num_classes=num_classes, average=None),

        "auroc_macro": MulticlassAUROC(num_classes=num_classes, average="macro"),
        "auprc_macro": MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
        "auprc_per_class": MulticlassAveragePrecision(num_classes=num_classes, average=None),

        "specificity_per_class": MulticlassSpecificity(num_classes=num_classes, average=None),

        "ece": MulticlassCalibrationError(num_classes=num_classes, n_bins=15),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=num_classes),
    }

@torch.no_grad()
def compute_metrics(metrics: Dict[str, torch.nn.Module]) -> Dict[str, Any]:
    """Compute and reset metrics, returning a plain dict."""
    results = {}
    for name, metric in metrics.items():
        val = metric.compute()
        if name == "confusion_matrix":
            results[name] = val.detach().cpu().numpy()
        elif torch.is_tensor(val) and val.ndim > 0:
            results[name] = val.detach().cpu().tolist()
        else:
            results[name] = float(val)
        metric.reset()
    return results


@torch.no_grad()
def compute_classification_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics directly from logits and targets.
    """
    if num_classes is None:
        num_classes = int(logits.shape[-1])

    # Build metric modules on CPU
    metrics = build_classification_metrics(num_classes)

    logits_cpu = logits.detach().to("cpu")
    targets_cpu = targets.detach().to("cpu").long()

    probs = torch.softmax(logits_cpu, dim=-1)
    prob_metrics = {"auroc_macro", "auprc_macro", "auprc_per_class", "ece"}

    for name, m in metrics.items():
        if name in prob_metrics:
            m.update(probs, targets_cpu)
        else:
            m.update(logits_cpu, targets_cpu)

    return compute_metrics(metrics)
