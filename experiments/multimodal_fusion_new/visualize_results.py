"""
Visualization script for reliability-switch experiment results.

Generates plots comparing fusion variants under stress tests.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

# Styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 350,
    "savefig.dpi": 350,
    "axes.linewidth": 1.2,
})

# Color scheme
VARIANT_COLORS = {
    "glrx": "#F6BD16",        # Golden
    "uniform_avg": "#5B8FF9",  # Blue
    "concat_mlp": "#61DDAA",   # Green
}

VARIANT_NAMES = {
    "glrx": "GLR-X",
    "uniform_avg": "Uniform Avg",
    "concat_mlp": "Concat + MLP",
}


def plot_stress_test_comparison(
    results_dir: str,
    output_dir: str,
    variants: Optional[List[str]] = None,
    metric: str = "f1_macro",
    split_filter: Optional[str] = None,
    head_filter: Optional[str] = None,
):
    """
    Plot stress test comparison across variants.
    
    Args:
        results_dir: Base results directory (e.g., experiments/multimodal_fusion_reliability_results)
        output_dir: Where to save plots
        variants: List of variant names to compare
        metric: Metric to plot
        split_filter: Filter by split (individual, group)
        head_filter: Filter by head (Engagement, Valence)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if variants is None:
        variants = ["glrx", "uniform_avg", "concat_mlp"]
    
    # Load aggregated results for each variant
    all_data = []
    
    for variant in variants:
        summary_path = Path(results_dir) / "kfold" / variant / "aggregated" / "stress_test_summary.csv"
        
        if not summary_path.exists():
            print(f"Warning: Summary not found for {variant} at {summary_path}")
            continue
        
        df = pd.read_csv(summary_path)
        df['variant'] = variant
        all_data.append(df)
    
    if not all_data:
        print("No data found to plot")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter
    combined = combined[combined['metric'] == metric]
    
    if split_filter:
        combined = combined[combined['split'] == split_filter]
    
    if head_filter:
        combined = combined[combined['head'] == head_filter]
    
    if combined.empty:
        print("No data after filtering")
        return
    
    # Plot each corruption type
    corruption_types = combined['corruption_type'].unique()
    corruption_types = [c for c in corruption_types if c in ['dropout', 'noise', 'shuffle']]
    
    n_panels = len(corruption_types)
    if n_panels == 0:
        print("No recognized corruption types found")
        return
    
    fig, axes = plt.subplots(1, n_panels, figsize=(18, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]
    
    fig.patch.set_facecolor("#FAFAFB")
    
    for ax, corr_type in zip(axes, corruption_types):
        ax.set_facecolor("#F4F6FA")
        ax.set_axisbelow(True)
        
        df_corr = combined[combined['corruption_type'] == corr_type]
        
        # Plot uniform corruption only (modality == 'all')
        df_corr = df_corr[df_corr['modality'] == 'all']
        
        for variant in variants:
            df_var = df_corr[df_corr['variant'] == variant]
            
            if df_var.empty:
                continue
            
            # Extract mean and std
            xs = df_var['corruption_param'].values
            means = df_var['mean'].values
            stds = df_var['std'].values
            
            # Sort by corruption_param
            order = np.argsort(xs)
            xs = xs[order]
            means = means[order]
            stds = stds[order]
            
            color = VARIANT_COLORS.get(variant, "#999999")
            label = VARIANT_NAMES.get(variant, variant)
            
            # Plot line
            ax.plot(xs, means, color=color, linewidth=2.5, label=label, zorder=3)
            
            # Plot points
            ax.scatter(xs, means, color=color, s=60, edgecolors='white', linewidths=0.8, zorder=4)
            
            # Error bars
            ax.errorbar(
                xs, means, yerr=stds,
                color=color, alpha=0.3, linewidth=1.0,
                capsize=4, capthick=1.0, zorder=2
            )
        
        # Styling
        if corr_type == "dropout":
            ax.set_title("Modality Dropout", pad=8, color="#1A1A1A")
            ax.set_xlabel("Dropout Probability", fontsize=11)
        elif corr_type == "noise":
            ax.set_title("Modality Noise", pad=8, color="#1A1A1A")
            ax.set_xlabel("Noise Scale (σ)", fontsize=11)
        elif corr_type == "shuffle":
            ax.set_title("Modality Shuffle", pad=8, color="#1A1A1A")
            ax.set_xlabel("Shuffle Probability", fontsize=11)
        
        # Grid and spines
        ax.yaxis.grid(True, linestyle=":", linewidth=0.9, alpha=0.7, color="#BFC7D5")
        ax.xaxis.grid(False)
        
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#A9B2C3")
    
    # Y-label
    ylabel = f"{metric.upper().replace('_', ' ')}"
    if split_filter:
        ylabel += f" ({split_filter.capitalize()})"
    if head_filter:
        ylabel += f" - {head_filter}"
    
    axes[0].set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(variants),
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=3.0,
        handlelength=2.0,
        fontsize=11,
    )
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.18, wspace=0.15)
    
    # Save
    suffix = f"{metric}"
    if split_filter:
        suffix += f"_{split_filter}"
    if head_filter:
        suffix += f"_{head_filter}"
    
    png_path = os.path.join(output_dir, f"reliability_comparison_{suffix}.png")
    pdf_path = os.path.join(output_dir, f"reliability_comparison_{suffix}.pdf")
    
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    
    plt.close(fig)


def plot_allocation_under_corruption(
    results_dir: str,
    output_dir: str,
    variant: str = "glrx",
):
    """
    Plot GLR-X allocation weights under increasing corruption.
    
    Args:
        results_dir: Base results directory
        output_dir: Where to save plots
        variant: Variant name (should support allocation tracking)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = Path(results_dir) / "kfold" / variant / "aggregated" / "stress_test_summary.csv"
    
    if not summary_path.exists():
        print(f"Summary not found at {summary_path}")
        return
    
    df = pd.read_csv(summary_path)
    
    # Filter for allocation weights
    df_alloc = df[df['metric'] == 'allocation_weight']
    
    if df_alloc.empty:
        print("No allocation data found")
        return
    
    # Branches (heads correspond to branch names for allocation)
    branches = df_alloc['head'].unique()
    corruption_types = df_alloc['corruption_type'].unique()
    corruption_types = [c for c in corruption_types if c in ['dropout', 'noise', 'shuffle']]
    
    # Filter for per-modality corruption
    df_alloc = df_alloc[df_alloc['modality'] != 'all']
    
    n_panels = len(corruption_types)
    if n_panels == 0:
        print("No corruption types found")
        return
    
    fig, axes = plt.subplots(1, n_panels, figsize=(18, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]
    
    fig.patch.set_facecolor("#FAFAFB")
    
    branch_colors = {
        "Videokinetic": "#5B8FF9",
        "Dialogue": "#61DDAA",
        "Acoustic": "#F6BD16",
    }
    
    for ax, corr_type in zip(axes, corruption_types):
        ax.set_facecolor("#F4F6FA")
        ax.set_axisbelow(True)
        
        df_corr = df_alloc[df_alloc['corruption_type'] == corr_type]
        
        for branch in branches:
            df_branch = df_corr[df_corr['head'] == branch]
            
            if df_branch.empty:
                continue
            
            xs = df_branch['corruption_param'].values
            means = df_branch['mean'].values
            stds = df_branch['std'].values
            
            order = np.argsort(xs)
            xs = xs[order]
            means = means[order]
            stds = stds[order]
            
            color = branch_colors.get(branch, "#999999")
            
            ax.plot(xs, means, color=color, linewidth=2.5, label=branch, zorder=3)
            ax.scatter(xs, means, color=color, s=60, edgecolors='white', linewidths=0.8, zorder=4)
            ax.errorbar(
                xs, means, yerr=stds,
                color=color, alpha=0.3, linewidth=1.0,
                capsize=4, capthick=1.0, zorder=2
            )
        
        # Styling
        if corr_type == "dropout":
            ax.set_title("Allocation under Dropout", pad=8, color="#1A1A1A")
            ax.set_xlabel("Dropout Probability", fontsize=11)
        elif corr_type == "noise":
            ax.set_title("Allocation under Noise", pad=8, color="#1A1A1A")
            ax.set_xlabel("Noise Scale", fontsize=11)
        elif corr_type == "shuffle":
            ax.set_title("Allocation under Shuffle", pad=8, color="#1A1A1A")
            ax.set_xlabel("Shuffle Probability", fontsize=11)
        
        ax.yaxis.grid(True, linestyle=":", linewidth=0.9, alpha=0.7, color="#BFC7D5")
        ax.xaxis.grid(False)
        
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#A9B2C3")
    
    axes[0].set_ylabel("Allocation Weight", fontsize=12, fontweight='bold')
    
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(branches),
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=3.0,
        handlelength=2.0,
        fontsize=11,
    )
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.18, wspace=0.15)
    
    png_path = os.path.join(output_dir, f"allocation_under_corruption_{variant}.png")
    pdf_path = os.path.join(output_dir, f"allocation_under_corruption_{variant}.pdf")
    
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize reliability-switch experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/multimodal_fusion_reliability_results",
        help="Base results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/viz_results/reliability_switch",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="glrx,uniform_avg,concat_mlp",
        help="Comma-separated list of variants to compare"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1_macro",
        help="Metric to plot"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Filter by split (individual, group)"
    )
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Filter by head (Engagement, Valence)"
    )
    parser.add_argument(
        "--plot-allocation",
        action="store_true",
        help="Also plot allocation tracking for GLR-X"
    )
    
    args = parser.parse_args()
    
    variants = [v.strip() for v in args.variants.split(',')]
    
    print("Plotting stress test comparison...")
    plot_stress_test_comparison(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        variants=variants,
        metric=args.metric,
        split_filter=args.split,
        head_filter=args.head,
    )
    
    if args.plot_allocation and 'glrx' in variants:
        print("\nPlotting allocation tracking...")
        plot_allocation_under_corruption(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            variant='glrx',
        )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

