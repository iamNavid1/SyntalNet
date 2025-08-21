from __future__ import annotations

from typing import Dict, Optional

import torch
import numpy as np
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
)


def build_classification_metrics(num_classes: int) -> Dict[str, torch.nn.Module]:
    """Create a dictionary of classification metrics for ``num_classes`` classes."""

    return {
        "accuracy": MulticlassAccuracy(num_classes=num_classes),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="micro"),
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "auroc_ovr": MulticlassAUROC(num_classes=num_classes, average="macro"),
        "auprc_macro": MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
        "ece": MulticlassCalibrationError(num_classes=num_classes, n_bins=15),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=num_classes),
    }

@torch.no_grad()
def compute_metrics(metrics: Dict[str, torch.nn.Module]) -> Dict[str, float]:
    """Compute and reset metrics, returning a plain dict of floats."""
    results = {}
    for name, metric in metrics.items():
        if name == "confusion_matrix":
            # Confusion matrix returns a tensor, convert to numpy for storage
            cm = metric.compute()
            results[name] = cm.cpu().numpy()
        else:
            results[name] = metric.compute().item()
        metric.reset()
    return results
