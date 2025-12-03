"""
Model Parameter Explorer
========================
Script to load CNN and LSTM models, count parameters, and experiment with increasing model sizes
while keeping input data dimensions the same.

USAGE:
    Activate your PyTorch environment first, then run:
    python model_param_explorer.py

If you get import errors, use model_param_calculator.py instead (doesn't require PyTorch).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from engine.utils import BRANCH_MODALITY_MAP
from models.TemporalCNN import TemporalCNN, FEATURE_DIMS as CNN_FEATURE_DIMS
from models.TemporalBiLSTM import TemporalBiLSTM, FEATURE_DIMS as LSTM_FEATURE_DIMS
from utils.model_stats import log_param_counts_detailed


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model_name, model, config):
    """Print a minimal summary of the model parameters."""
    total, trainable = count_parameters(model)
    print(f"{model_name}: total_params={total:,}, trainable_params={trainable:,}")


class _StdoutLogger:
    """Minimal logger interface for model_stats helpers."""

    def info(self, msg: str, *args, **kwargs) -> None:
        if args:
            msg = msg % args
        print(msg)


def print_model_structure(model: nn.Module, max_depth: int = 3) -> None:
    """
    Print a compact tree of the model with parameter counts per module.

    This is a lightweight wrapper around utils.model_stats.log_param_counts_detailed.
    """
    logger = _StdoutLogger()
    log_param_counts_detailed(
        model,
        logger,
        trainable_only=True,
        max_depth=max_depth,
        min_params=0,
        topk_per_level=10,
        include_name_regex=None,
        exclude_name_regex=None,
        show_leaf_shapes=False,
        show_type=True,
    )


def build_dummy_batch(
    feature_dims: Dict[str, int],
    seq_len: int = 8,
    batch_size: int = 1,
    num_persons: int = 1,
    emb_dim: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Build a minimal dummy batch that matches the expected input structure
    for TemporalCNN / TemporalBiLSTM so that all lazy submodules are built.
    """
    if device is None:
        device = torch.device("cpu")

    B, P, T = batch_size, num_persons, seq_len
    batch_data: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

    for branch, spec in BRANCH_MODALITY_MAP.items():
        mod_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Feature modalities (use real feature dimensions)
        for modality in spec["feat"]:
            dim = feature_dims[modality]
            x = torch.randn(B, P, T, dim, device=device)
            m = torch.ones(B, P, T, 1, device=device)
            mod_dict[modality] = (x, m)

        # Embedding modality (dimension is arbitrary but fixed)
        emb_mod = spec["emb"]
        x_emb = torch.randn(B, P, T, emb_dim, device=device)
        m_emb = torch.ones(B, P, T, 1, device=device)
        mod_dict[emb_mod] = (x_emb, m_emb)

        batch_data[branch] = mod_dict

    return batch_data


def explore_lstm_sizes():
    """Create a single BiLSTM with default-style hyperparameters and report params."""
    config = {
        "hidden_size": 96,
        "num_lstm_layers": 1,
        "classifier_hidden": 96,
        "dropout": 0.2,
        "bidirectional": True,
    }

    try:
        model = TemporalBiLSTM(
            hidden_size=config["hidden_size"],
            num_lstm_layers=config["num_lstm_layers"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
            ind_cls_heads=("Engagement", "Lead"),
            grp_cls_heads=("Synchrony", "Confidence", "Transition"),
            classifier_hidden=config["classifier_hidden"],
        )

        # Run a dummy forward pass so that all lazy LSTM encoders are built
        dummy_batch = build_dummy_batch(LSTM_FEATURE_DIMS)
        with torch.no_grad():
            model(dummy_batch)

        print_model_summary("BiLSTM (default)", model, config)
        # print("\nBiLSTM structure (truncated):")
        # print_model_structure(model, max_depth=3)
        total, _ = count_parameters(model)
        return {"Baseline (Default)": total}
    except Exception as e:
        print(f"[ERROR] Failed to create BiLSTM default configuration: {e}")
        return {}


def explore_cnn_sizes():
    """Create a single CNN with default-style hyperparameters and report params."""
    config = {
        "hidden_size": 128,
        "num_cnn_layers": 3,
        "kernel_size": 3,
        "classifier_hidden": 128,
        "dropout": 0.2,
    }

    try:
        model = TemporalCNN(
            hidden_size=config["hidden_size"],
            num_cnn_layers=config["num_cnn_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
            ind_cls_heads=("Engagement", "Lead"),
            grp_cls_heads=("Synchrony", "Confidence", "Transition"),
            classifier_hidden=config["classifier_hidden"],
        )

        # Run a dummy forward pass so that all lazy CNN encoders are built
        dummy_batch = build_dummy_batch(CNN_FEATURE_DIMS)
        with torch.no_grad():
            model(dummy_batch)

        print_model_summary("CNN (default)", model, config)
        # print("\nCNN structure (truncated):")
        # print_model_structure(model, max_depth=3)
        total, _ = count_parameters(model)
        return {"Baseline (Default)": total}
    except Exception as e:
        print(f"[ERROR] Failed to create CNN default configuration: {e}")
        return {}



def main():
    """Main function to run parameter counts for default models."""
    print("\nModel parameter counts (default configurations):")
    explore_lstm_sizes()
    explore_cnn_sizes()


if __name__ == "__main__":
    main()

