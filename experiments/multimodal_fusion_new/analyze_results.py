"""
Analysis script for reliability-switch experiment results.

Generates summary tables and statistics comparing fusion variants.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional


def load_variant_results(
    results_dir: str,
    variant: str,
    cv_mode: str = "kfold"
) -> pd.DataFrame:
    """Load aggregated results for a variant."""
    
    summary_path = Path(results_dir) / cv_mode / variant / "aggregated" / "stress_test_summary.csv"
    
    if not summary_path.exists():
        print(f"Warning: Summary not found for {variant} at {summary_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(summary_path)
    df['variant'] = variant
    return df


def compare_clean_performance(
    results_dir: str,
    variants: List[str],
    cv_mode: str = "kfold",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare clean (uncorrupted) performance across variants.
    
    Returns:
        DataFrame with columns: variant, split, head, metric, mean, std
    """
    
    all_clean = []
    
    for variant in variants:
        clean_path = Path(results_dir) / cv_mode / variant / "aggregated" / "clean_metrics_all_folds.csv"
        
        if not clean_path.exists():
            print(f"Warning: Clean metrics not found for {variant}")
            continue
        
        df = pd.read_csv(clean_path)
        
        # Extract per-head metrics
        metric_cols = [c for c in df.columns if c.endswith(('_f1', '_auroc', '_auprc'))]
        
        for col in metric_cols:
            parts = col.rsplit('_', 1)
            prefix = parts[0]  # e.g., "individual_Engagement"
            metric_name = parts[1]  # e.g., "f1"
            
            split_head = prefix.split('_', 1)
            if len(split_head) == 2:
                split, head = split_head
                
                all_clean.append({
                    'variant': variant,
                    'split': split,
                    'head': head,
                    'metric': metric_name,
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                })
    
    if not all_clean:
        print("No clean performance data found")
        return pd.DataFrame()
    
    result = pd.DataFrame(all_clean)
    
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved clean performance comparison to: {output_path}")
    
    return result


def compare_robustness(
    results_dir: str,
    variants: List[str],
    corruption_type: str = "dropout",
    metric: str = "f1_macro",
    split_filter: Optional[str] = None,
    head_filter: Optional[str] = None,
    cv_mode: str = "kfold",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare robustness (performance degradation) across variants.
    
    Returns:
        DataFrame with robustness metrics per variant
    """
    
    all_data = []
    
    for variant in variants:
        df = load_variant_results(results_dir, variant, cv_mode)
        
        if df.empty:
            continue
        
        # Filter
        df = df[df['corruption_type'] == corruption_type]
        df = df[df['metric'] == metric]
        df = df[df['modality'] == 'all']  # Uniform corruption
        
        if split_filter:
            df = df[df['split'] == split_filter]
        if head_filter:
            df = df[df['head'] == head_filter]
        
        if df.empty:
            continue
        
        # Get clean (0.0) and max corruption performance
        clean_row = df[df['corruption_param'] == 0.0]
        max_corr_row = df[df['corruption_param'] == df['corruption_param'].max()]
        
        if not clean_row.empty and not max_corr_row.empty:
            clean_perf = clean_row['mean'].values[0]
            max_corr_perf = max_corr_row['mean'].values[0]
            degradation = clean_perf - max_corr_perf
            relative_degradation = degradation / clean_perf if clean_perf > 0 else 0
            
            # Compute slope (linear fit)
            xs = df['corruption_param'].values
            ys = df['mean'].values
            
            if len(xs) > 1:
                coeffs = np.polyfit(xs, ys, 1)
                slope = coeffs[0]
            else:
                slope = 0.0
            
            all_data.append({
                'variant': variant,
                'corruption_type': corruption_type,
                'metric': metric,
                'split': split_filter or 'all',
                'head': head_filter or 'all',
                'clean_performance': clean_perf,
                'max_corrupted_performance': max_corr_perf,
                'absolute_degradation': degradation,
                'relative_degradation': relative_degradation,
                'degradation_slope': slope,
            })
    
    if not all_data:
        print("No robustness data found")
        return pd.DataFrame()
    
    result = pd.DataFrame(all_data)
    
    # Sort by relative degradation (lower is better = more robust)
    result = result.sort_values('relative_degradation')
    
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved robustness comparison to: {output_path}")
    
    return result


def generate_summary_table(
    results_dir: str,
    variants: List[str],
    cv_mode: str = "kfold",
    output_dir: Optional[str] = None,
):
    """
    Generate comprehensive summary tables.
    """
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SUMMARY: Clean Performance")
    print("=" * 80)
    
    clean_perf = compare_clean_performance(
        results_dir, variants, cv_mode,
        output_path=os.path.join(output_dir, "clean_performance.csv") if output_dir else None
    )
    
    if not clean_perf.empty:
        # Pivot for better display
        pivot = clean_perf.pivot_table(
            index=['split', 'head', 'metric'],
            columns='variant',
            values='mean'
        )
        print(pivot.to_string())
        print()
    
    print("=" * 80)
    print("SUMMARY: Robustness to Corruption")
    print("=" * 80)
    
    for corr_type in ['dropout', 'noise', 'shuffle']:
        print(f"\n{corr_type.upper()}:")
        print("-" * 40)
        
        robustness = compare_robustness(
            results_dir, variants, corr_type, metric='f1_macro',
            cv_mode=cv_mode,
            output_path=os.path.join(output_dir, f"robustness_{corr_type}.csv") if output_dir else None
        )
        
        if not robustness.empty:
            # Print key columns
            cols = ['variant', 'clean_performance', 'max_corrupted_performance', 
                    'relative_degradation', 'degradation_slope']
            print(robustness[cols].to_string(index=False))
            print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze reliability-switch experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/multimodal_fusion_reliability_results",
        help="Base results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/multimodal_fusion_reliability_results/analysis",
        help="Output directory for analysis tables"
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="glrx,uniform_avg,concat_mlp",
        help="Comma-separated list of variants to analyze"
    )
    parser.add_argument(
        "--cv-mode",
        type=str,
        default="kfold",
        help="Cross-validation mode"
    )
    
    args = parser.parse_args()
    
    variants = [v.strip() for v in args.variants.split(',')]
    
    generate_summary_table(
        results_dir=args.results_dir,
        variants=variants,
        cv_mode=args.cv_mode,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

