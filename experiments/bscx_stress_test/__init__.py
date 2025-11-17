"""
Stress test evaluation suite for multi-channel fusion modules.

This package provides tools for systematically evaluating model robustness
under various corruption scenarios.
"""

from experiments.bscx_stress_test.runner import run_stress_test
from experiments.bscx_stress_test.evaluator import StressTestEvaluator
from experiments.bscx_stress_test.metrics_collector import MetricsCollector
from experiments.bscx_stress_test.model_loader import load_model_with_fusion_type

__all__ = [
    "run_stress_test",
    "StressTestEvaluator",
    "MetricsCollector",
    "load_model_with_fusion_type",
]

