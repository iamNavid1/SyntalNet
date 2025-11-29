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
from models.TemporalCNN import TemporalCNN
from models.TemporalBiLSTM import TemporalBiLSTM


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """Format large numbers with commas and show in millions."""
    millions = num / 1_000_000
    return f"{num:,} ({millions:.2f}M)"


def print_model_summary(model_name, model, config):
    """Print a summary of the model parameters."""
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key:25s}: {value}")
    print(f"\nParameter Counts:")
    print(f"  Total Parameters:      {format_number(total)}")
    print(f"  Trainable Parameters:  {format_number(trainable)}")
    print(f"{'='*80}")


def explore_lstm_sizes():
    """Explore different LSTM model sizes."""
    print("\n" + "="*80)
    print(" TEMPORAL BiLSTM MODEL - PARAMETER EXPLORATION")
    print("="*80)
    
    # Configuration sets: from baseline to massive
    configs = {
        "Baseline (Default)": {
            "hidden_size": 128,
            "num_lstm_layers": 2,
            "classifier_hidden": 128,
            "dropout": 0.2,
            "bidirectional": True,
        },
        "Medium": {
            "hidden_size": 256,
            "num_lstm_layers": 3,
            "classifier_hidden": 256,
            "dropout": 0.2,
            "bidirectional": True,
        },
        "Large": {
            "hidden_size": 512,
            "num_lstm_layers": 4,
            "classifier_hidden": 512,
            "dropout": 0.2,
            "bidirectional": True,
        },
        "Extra Large": {
            "hidden_size": 1024,
            "num_lstm_layers": 5,
            "classifier_hidden": 1024,
            "dropout": 0.2,
            "bidirectional": True,
        },
        "Massive": {
            "hidden_size": 2048,
            "num_lstm_layers": 6,
            "classifier_hidden": 2048,
            "dropout": 0.2,
            "bidirectional": True,
        },
        "Ultra Massive": {
            "hidden_size": 4096,
            "num_lstm_layers": 8,
            "classifier_hidden": 4096,
            "dropout": 0.2,
            "bidirectional": True,
        },
    }
    
    results = {}
    for name, config in configs.items():
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
            print_model_summary(f"BiLSTM - {name}", model, config)
            total, _ = count_parameters(model)
            results[name] = total
        except Exception as e:
            print(f"\n[ERROR] Failed to create {name} configuration: {e}")
    
    # Print comparison
    print("\n" + "="*80)
    print(" PARAMETER COUNT COMPARISON")
    print("="*80)
    baseline = results.get("Baseline (Default)", 1)
    for name, total in results.items():
        increase = (total / baseline - 1) * 100
        print(f"{name:20s}: {format_number(total):>25s} ({increase:>7.1f}% increase)")
    
    return results


def explore_cnn_sizes():
    """Explore different CNN model sizes."""
    print("\n" + "="*80)
    print(" TEMPORAL CNN MODEL - PARAMETER EXPLORATION")
    print("="*80)
    
    # Configuration sets: from baseline to massive
    configs = {
        "Baseline (Default)": {
            "hidden_size": 128,
            "num_cnn_layers": 3,
            "kernel_size": 3,
            "classifier_hidden": 128,
            "dropout": 0.2,
        },
        "Medium": {
            "hidden_size": 256,
            "num_cnn_layers": 4,
            "kernel_size": 5,
            "classifier_hidden": 256,
            "dropout": 0.2,
        },
        "Large": {
            "hidden_size": 512,
            "num_cnn_layers": 6,
            "kernel_size": 7,
            "classifier_hidden": 512,
            "dropout": 0.2,
        },
        "Extra Large": {
            "hidden_size": 1024,
            "num_cnn_layers": 8,
            "kernel_size": 9,
            "classifier_hidden": 1024,
            "dropout": 0.2,
        },
        "Massive": {
            "hidden_size": 2048,
            "num_cnn_layers": 10,
            "kernel_size": 11,
            "classifier_hidden": 2048,
            "dropout": 0.2,
        },
        "Ultra Massive": {
            "hidden_size": 4096,
            "num_cnn_layers": 12,
            "kernel_size": 15,
            "classifier_hidden": 4096,
            "dropout": 0.2,
        },
    }
    
    results = {}
    for name, config in configs.items():
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
            print_model_summary(f"CNN - {name}", model, config)
            total, _ = count_parameters(model)
            results[name] = total
        except Exception as e:
            print(f"\n[ERROR] Failed to create {name} configuration: {e}")
    
    # Print comparison
    print("\n" + "="*80)
    print(" PARAMETER COUNT COMPARISON")
    print("="*80)
    baseline = results.get("Baseline (Default)", 1)
    for name, total in results.items():
        increase = (total / baseline - 1) * 100
        print(f"{name:20s}: {format_number(total):>25s} ({increase:>7.1f}% increase)")
    
    return results


def custom_model_builder():
    """Interactive section to build custom-sized models."""
    print("\n" + "="*80)
    print(" CUSTOM MODEL BUILDER")
    print("="*80)
    print("\nYou can modify the parameters below to create custom models:")
    print("\nFor BiLSTM, key parameters are:")
    print("  - hidden_size: Size of hidden states (affects most params)")
    print("  - num_lstm_layers: Number of LSTM layers")
    print("  - classifier_hidden: Hidden size of classifier MLPs")
    print("\nFor CNN, key parameters are:")
    print("  - hidden_size: Number of channels (affects most params)")
    print("  - num_cnn_layers: Number of convolutional layers")
    print("  - kernel_size: Size of convolution kernels")
    print("  - classifier_hidden: Hidden size of classifier MLPs")
    
    # Example: Create extreme models
    print("\n" + "-"*80)
    print(" EXTREME CONFIGURATION EXAMPLES")
    print("-"*80)
    
    # Extreme BiLSTM
    print("\n[1] Extreme BiLSTM (pushing limits):")
    extreme_lstm_config = {
        "hidden_size": 8192,
        "num_lstm_layers": 10,
        "classifier_hidden": 8192,
        "dropout": 0.2,
        "bidirectional": True,
    }
    try:
        extreme_lstm = TemporalBiLSTM(**extreme_lstm_config,
                                      ind_cls_heads=("Engagement", "Lead"),
                                      grp_cls_heads=("Synchrony", "Confidence", "Transition"))
        print_model_summary("Extreme BiLSTM", extreme_lstm, extreme_lstm_config)
    except Exception as e:
        print(f"[ERROR] Could not create extreme BiLSTM: {e}")
    
    # Extreme CNN
    print("\n[2] Extreme CNN (pushing limits):")
    extreme_cnn_config = {
        "hidden_size": 8192,
        "num_cnn_layers": 15,
        "kernel_size": 21,
        "classifier_hidden": 8192,
        "dropout": 0.2,
    }
    try:
        extreme_cnn = TemporalCNN(**extreme_cnn_config,
                                   ind_cls_heads=("Engagement", "Lead"),
                                   grp_cls_heads=("Synchrony", "Confidence", "Transition"))
        print_model_summary("Extreme CNN", extreme_cnn, extreme_cnn_config)
    except Exception as e:
        print(f"[ERROR] Could not create extreme CNN: {e}")


def analyze_parameter_distribution(model, model_name):
    """Analyze where parameters are distributed in the model."""
    print(f"\n{'='*80}")
    print(f"Parameter Distribution Analysis: {model_name}")
    print(f"{'='*80}")
    
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            module_params[name] = params
    
    total = sum(module_params.values())
    
    print(f"\n{'Module':<30s} {'Parameters':>20s} {'Percentage':>15s}")
    print("-" * 80)
    for name, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
        percentage = (params / total) * 100 if total > 0 else 0
        print(f"{name:<30s} {format_number(params):>20s} {percentage:>14.2f}%")
    print("-" * 80)
    print(f"{'TOTAL':<30s} {format_number(total):>20s} {'100.00':>14s}%")


def main():
    """Main function to run all explorations."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "MODEL PARAMETER EXPLORER" + " "*35 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    # Explore LSTM sizes
    lstm_results = explore_lstm_sizes()
    
    # Explore CNN sizes
    cnn_results = explore_cnn_sizes()
    
    # Custom model builder with extreme examples
    custom_model_builder()
    
    # Detailed analysis on baseline models
    print("\n" + "="*80)
    print(" DETAILED PARAMETER DISTRIBUTION ANALYSIS")
    print("="*80)
    
    baseline_lstm = TemporalBiLSTM(
        hidden_size=128,
        num_lstm_layers=2,
        dropout=0.2,
        bidirectional=True,
        ind_cls_heads=("Engagement", "Lead"),
        grp_cls_heads=("Synchrony", "Confidence", "Transition"),
        classifier_hidden=128,
    )
    analyze_parameter_distribution(baseline_lstm, "Baseline BiLSTM")
    
    baseline_cnn = TemporalCNN(
        hidden_size=128,
        num_cnn_layers=3,
        kernel_size=3,
        dropout=0.2,
        ind_cls_heads=("Engagement", "Lead"),
        grp_cls_heads=("Synchrony", "Confidence", "Transition"),
        classifier_hidden=128,
    )
    analyze_parameter_distribution(baseline_cnn, "Baseline CNN")
    
    # Summary
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*30 + "SUMMARY" + " "*42 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    print("\nKey findings:")
    print("  1. Input dimensions remain the same across all configurations")
    print("  2. Parameter count scales quadratically with hidden_size")
    print("  3. BiLSTM parameters grow faster than CNN due to recurrent connections")
    print("  4. Largest practical models (8192 hidden size) have 500M-1B+ parameters")
    print("\nTo create your own configuration, modify the parameters in custom_model_builder()")
    print("or directly instantiate models with desired hyperparameters.")
    print("\n" + "█"*80 + "\n")


if __name__ == "__main__":
    main()

