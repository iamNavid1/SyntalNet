"""
Example workflow demonstrating the complete reliability-switch experiment pipeline.

This script shows how to programmatically run experiments, analyze results,
and generate visualizations.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pandas as pd

from experiments.multimodal_fusion_new.config import load_config, get_base_checkpoint_path, get_fold_dirs
from experiments.multimodal_fusion_new.model_loader import load_checkpoint_for_reliability_training
from experiments.multimodal_fusion_new.fusion_variants import get_variant_config
import models.builders as build


def example_1_load_frozen_model():
    """Example 1: Load a frozen SyntalNet model with new fusion variant."""
    
    print("=" * 80)
    print("EXAMPLE 1: Loading frozen SyntalNet with GLR-X fusion")
    print("=" * 80)
    
    # Load experiment config
    config_path = "experiments/multimodal_fusion_new/configs/reliability_glrx.yaml"
    exp_config = load_config(config_path)
    
    # Load base SyntalNet config
    base_config = build.config(exp_config["base_config_path"])
    
    # Get checkpoint path for fold 0
    fold_idx = 0
    checkpoint_path = get_base_checkpoint_path(
        exp_config["base_checkpoint_dir"],
        fold_idx,
        cv_mode="kfold"
    )
    
    print(f"\nBase checkpoint: {checkpoint_path}")
    
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model, _ = load_checkpoint_for_reliability_training(
        checkpoint_path=checkpoint_path,
        config=base_config,
        fusion_variant="glrx",
        fusion_kwargs=exp_config["fusion_kwargs"],
        device=device,
    )
    
    print(f"\nModel loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Fusion type: {type(model.mm_fusion).__name__}")
    
    # Verify frozen parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    return model


def example_2_test_corruptions():
    """Example 2: Test corruption functions on dummy data."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Testing corruption functions")
    print("=" * 80)
    
    from experiments.multimodal_fusion_new.corruptions import (
        apply_reliability_switch_corruption,
        apply_uniform_corruption,
        corrupt_branch_dropout,
        corrupt_branch_noise,
    )
    
    # Create dummy branch embeddings
    B, D = 16, 128
    z_video = torch.randn(B, D)
    z_audio = torch.randn(B, D)
    z_text = torch.randn(B, D)
    
    z_list = [z_video, z_audio, z_text]
    
    print(f"\nOriginal embeddings shape: {z_video.shape}")
    print(f"Number of branches: {len(z_list)}")
    
    # Test reliability-switch corruption
    print("\nApplying reliability-switch corruption...")
    z_corrupted = apply_reliability_switch_corruption(
        z_list,
        p_corrupt=0.7,
        corruption_strengths={
            'dropout': (0.1, 0.5),
            'noise': (0.1, 0.5),
            'shuffle': (0.1, 0.5),
        }
    )
    
    # Check differences
    for i, (z_orig, z_corr) in enumerate(zip(z_list, z_corrupted)):
        diff = (z_orig - z_corr).abs().mean().item()
        print(f"  Branch {i}: mean absolute difference = {diff:.4f}")
    
    # Test uniform dropout
    print("\nApplying uniform dropout (0.3)...")
    z_dropout = apply_uniform_corruption(z_list, "dropout", 0.3)
    
    # Test noise
    print("Applying uniform noise (0.5)...")
    z_noise = apply_uniform_corruption(z_list, "noise", 0.5)
    
    print("\nCorruption tests complete!")


def example_3_analyze_results():
    """Example 3: Load and analyze experiment results."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Analyzing experiment results")
    print("=" * 80)
    
    from experiments.multimodal_fusion_new.analyze_results import (
        compare_clean_performance,
        compare_robustness,
    )
    
    results_dir = "experiments/multimodal_fusion_reliability_results"
    variants = ["glrx", "uniform_avg", "concat_mlp"]
    
    # Check if results exist
    glrx_path = Path(results_dir) / "kfold" / "glrx" / "aggregated" / "stress_test_summary.csv"
    
    if not glrx_path.exists():
        print(f"\nResults not found at: {glrx_path}")
        print("Run experiments first using:")
        print("  bash experiments/multimodal_fusion_new/run_reliability_experiments.sh")
        return
    
    print("\nComparing clean performance...")
    clean_perf = compare_clean_performance(results_dir, variants)
    
    if not clean_perf.empty:
        print("\nClean Performance Summary:")
        pivot = clean_perf.pivot_table(
            index=['metric'],
            columns='variant',
            values='mean',
            aggfunc='mean'
        )
        print(pivot)
    
    print("\n\nComparing robustness to dropout...")
    robustness = compare_robustness(
        results_dir, variants,
        corruption_type="dropout",
        metric="f1_macro"
    )
    
    if not robustness.empty:
        print("\nRobustness Summary (Dropout):")
        print(robustness[['variant', 'clean_performance', 'relative_degradation']].to_string(index=False))
        
        # Find most robust
        best = robustness.iloc[0]
        print(f"\n✓ Most robust: {best['variant']} (degradation: {best['relative_degradation']:.2%})")


def example_4_generate_plots():
    """Example 4: Generate visualizations programmatically."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Generating visualizations")
    print("=" * 80)
    
    from experiments.multimodal_fusion_new.visualize_results import (
        plot_stress_test_comparison,
        plot_allocation_under_corruption,
    )
    
    results_dir = "experiments/multimodal_fusion_reliability_results"
    output_dir = "experiments/viz_results/reliability_switch"
    
    # Check if results exist
    glrx_path = Path(results_dir) / "kfold" / "glrx" / "aggregated" / "stress_test_summary.csv"
    
    if not glrx_path.exists():
        print(f"\nResults not found. Run experiments first.")
        return
    
    print("\nGenerating stress test comparison plots...")
    plot_stress_test_comparison(
        results_dir=results_dir,
        output_dir=output_dir,
        variants=["glrx", "uniform_avg", "concat_mlp"],
        metric="f1_macro",
    )
    
    print("\nGenerating allocation tracking plots...")
    plot_allocation_under_corruption(
        results_dir=results_dir,
        output_dir=output_dir,
        variant="glrx",
    )
    
    print(f"\nPlots saved to: {output_dir}")


def main():
    """Run all examples."""
    
    print("\n" + "=" * 80)
    print("RELIABILITY-SWITCH EXPERIMENT: Example Workflow")
    print("=" * 80)
    
    try:
        # Example 1: Load model
        model = example_1_load_frozen_model()
        
        # Example 2: Test corruptions
        example_2_test_corruptions()
        
        # Example 3: Analyze results (if available)
        example_3_analyze_results()
        
        # Example 4: Generate plots (if results available)
        example_4_generate_plots()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Example workflow complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run full experiments: bash experiments/multimodal_fusion_new/run_reliability_experiments.sh")
    print("2. Analyze results: python experiments/multimodal_fusion_new/analyze_results.py")
    print("3. Visualize: python experiments/multimodal_fusion_new/visualize_results.py --plot-allocation")
    print()


if __name__ == "__main__":
    main()

