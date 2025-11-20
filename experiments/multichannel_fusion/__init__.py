"""
Stress test evaluation suite for multi-channel fusion modules.

This package provides tools for systematically evaluating model robustness
under various corruption scenarios.
"""

from experiments.multichannel_fusion.runner import run_stress_test
from experiments.multichannel_fusion.evaluator import StressTestEvaluator
from experiments.multichannel_fusion.metrics_collector import MetricsCollector
from experiments.multichannel_fusion.model_loader import load_model_with_fusion_type

__all__ = [
    "run_stress_test",
    "StressTestEvaluator",
    "MetricsCollector",
    "load_model_with_fusion_type",
]

