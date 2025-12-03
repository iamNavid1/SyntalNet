from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------- Constants -----------------------------

# Label constructs in display order
CONSTRUCTS = [
    {"name": "Engagement", "key": "individual_Engagement", "short": "Engagement"},
    {"name": "Lead", "key": "individual_Lead", "short": "Lead"},
    {"name": "Synchrony", "key": "group_Synchrony", "short": "Synchrony"},
    {"name": "Confidence", "key": "group_Confidence", "short": "Confidence"},
    {"name": "Transition", "key": "group_Transition", "short": "Transition"},
]

# Data proportions for Experiment 1
EXP1_PROPORTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
EXP1_LABELS = ["Zero-shot", "5%", "10%", "15%", "20%", "25%"]

# Data proportions and epochs for Experiment 2
EXP2_PROPORTIONS = [0.05, 0.10, 0.15, 0.20, 0.25]
EXP2_EPOCHS = [2, 8, 12, 16, 20]

# Color palette for proportions
COLORS = {
    0.0: "#2C2C2C",    # Dark gray for zero-shot
    0.05: "#5B8FF9",   # Blue
    0.10: "#5AD8A6",   # Green
    0.15: "#5D7092",   # Purple-gray
    0.20: "#F6BD16",   # Yellow-orange
    0.25: "#E8684A",   # Red-orange
}


# ----------------------------- Argument Parsing -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="OOD Adaptation Experiment Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize OOD adaptation results
  python OOD_adaptation_results.py --input-dir experiments/ood_adaptation_results
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./experiments/ood_adaptation_results",
        help="Directory containing experiment CSV files (default: ./experiments/ood_adaptation_results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/viz_results/ood_adaptation",
        help="Directory to save figures (default: ./experiments/viz_results/ood_adaptation)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't show the plot (only save to file)"
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="ood_adaptation_results",
        help="Base name for output figure files (default: ood_adaptation_results)"
    )
    
    return parser.parse_args()


# ----------------------------- Data Loading -----------------------------

def load_exp1_data(input_dir: str) -> pd.DataFrame:
    """Load Experiment 1: Data-portion sweep aggregated results."""
    csv_path = os.path.join(input_dir, "exp1_data_portion_sweep_aggregated.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Experiment 1 CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_exp2_data(input_dir: str) -> pd.DataFrame:
    """Load Experiment 2: Epoch sweep aggregated results."""
    csv_path = os.path.join(input_dir, "exp2_epoch_sweep_aggregated.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Experiment 2 CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_exp3_data(input_dir: str) -> pd.DataFrame:
    """Load Experiment 3: Frozen backbone item split aggregated results."""
    csv_path = os.path.join(input_dir, "exp3_frozen_backbone_item_split_aggregated.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Experiment 3 CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


# ----------------------------- Data Extraction -----------------------------

def extract_construct_metrics(
    df: pd.DataFrame,
    construct_key: str,
    metric: str = "auprc_macro"
) -> Tuple[float, float]:
    """
    Extract mean and std for a specific construct and metric.
    
    Args:
        df: DataFrame row (single row expected)
        construct_key: Construct key (e.g., "group_Confidence")
        metric: Metric name (default: "auprc_macro")
    
    Returns:
        Tuple of (mean, std)
    """
    mean_col = f"{construct_key}_{metric}_mean"
    std_col = f"{construct_key}_{metric}_std"
    
    mean_val = df[mean_col].values[0] if mean_col in df.columns else np.nan
    std_val = df[std_col].values[0] if std_col in df.columns else np.nan
    
    return mean_val, std_val


def get_exp1_data_for_plotting(exp1_df: pd.DataFrame) -> Dict:
    """
    Extract Experiment 1 data for plotting.
    
    Returns:
        Dictionary with structure:
        {
            construct_name: {
                "proportions": [0.0, 0.05, 0.10, ...],
                "means": [mean_values],
                "stds": [std_values]
            }
        }
    """
    data = {}
    
    for construct in CONSTRUCTS:
        construct_key = construct["key"]
        construct_name = construct["short"]
        
        means = []
        stds = []
        
        for proportion in EXP1_PROPORTIONS:
            # Filter for this proportion
            rows = exp1_df[exp1_df["proportion"] == proportion]
            
            if len(rows) == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                # For zero-shot, use epoch 0; for others, use the last epoch (epoch 20)
                if proportion == 0.0:
                    row = rows[rows["epoch"] == 0]
                else:
                    row = rows[rows["epoch"] == 20]  # Use last epoch
                
                if len(row) == 0:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    mean, std = extract_construct_metrics(row, construct_key)
                    means.append(mean)
                    stds.append(std)
        
        data[construct_name] = {
            "proportions": EXP1_PROPORTIONS,
            "means": np.array(means),
            "stds": np.array(stds)
        }
    
    return data


def get_exp2_data_for_plotting(exp2_df: pd.DataFrame, exp1_df: pd.DataFrame) -> Dict:
    """
    Extract Experiment 2 data for plotting.
    
    Returns:
        Dictionary with structure:
        {
            proportion: {
                "epochs": [2, 8, 12, 16, 20],
                "means": [average AUPRC across all constructs],
                "stds": [std across all constructs]
            }
        }
        Also includes "zero_shot" key with mean and std.
    """
    data = {}
    
    # Get zero-shot data (from exp1)
    zero_shot_row = exp1_df[exp1_df["proportion"] == 0.0]
    zero_shot_means = []
    
    for construct in CONSTRUCTS:
        mean, _ = extract_construct_metrics(zero_shot_row, construct["key"])
        if not np.isnan(mean):
            zero_shot_means.append(mean)
    
    data["zero_shot"] = {
        "mean": np.mean(zero_shot_means) if zero_shot_means else np.nan,
        "std": np.std(zero_shot_means) if zero_shot_means else np.nan
    }
    
    # Get data for each proportion
    for proportion in EXP2_PROPORTIONS:
        means_per_epoch = []
        
        for epoch in EXP2_EPOCHS:
            # Filter for this proportion and epoch
            row = exp2_df[(exp2_df["proportion"] == proportion) & (exp2_df["epoch"] == epoch)]
            
            if len(row) == 0:
                means_per_epoch.append(np.nan)
            else:
                # Average across all constructs
                construct_means = []
                for construct in CONSTRUCTS:
                    mean, _ = extract_construct_metrics(row, construct["key"])
                    if not np.isnan(mean):
                        construct_means.append(mean)
                
                means_per_epoch.append(np.mean(construct_means) if construct_means else np.nan)
        
        data[proportion] = {
            "epochs": EXP2_EPOCHS,
            "means": np.array(means_per_epoch)
        }
    
    return data


def get_exp3_data_for_plotting(exp3_df: pd.DataFrame, exp1_df: pd.DataFrame) -> Dict:
    """
    Extract Experiment 3 data for plotting (last epoch only).
    
    Returns:
        Dictionary with structure:
        {
            construct_name: {
                "zero_shot": (mean, std),
                "frozen_backbone": (mean, std)
            }
        }
    """
    data = {}
    
    # Get last epoch from exp3
    last_epoch = exp3_df["epoch"].max()
    exp3_row = exp3_df[exp3_df["epoch"] == last_epoch]
    
    # Get zero-shot data (from exp1)
    zero_shot_row = exp1_df[exp1_df["proportion"] == 0.0]
    
    for construct in CONSTRUCTS:
        construct_key = construct["key"]
        construct_name = construct["short"]
        
        # Zero-shot
        zs_mean, zs_std = extract_construct_metrics(zero_shot_row, construct_key)
        
        # Frozen backbone
        fb_mean, fb_std = extract_construct_metrics(exp3_row, construct_key)
        
        data[construct_name] = {
            "zero_shot": (zs_mean, zs_std),
            "frozen_backbone": (fb_mean, fb_std)
        }
    
    return data


# ----------------------------- Visualization -----------------------------

def plot_experiment_1(ax, data: Dict):
    """
    Plot Experiment 1: Data-portion sweep.
    5 clusters (one per construct), each with 6 bars (zero-shot + 5 proportions).
    """
    n_constructs = len(CONSTRUCTS)
    n_bars = len(EXP1_PROPORTIONS)
    bar_width = 0.12
    inner_gap = 0.02
    group_gap = 0.5
    
    # Calculate x-positions for each bar
    x_positions = []
    for i in range(n_constructs):
        base = i * (n_bars * bar_width + (n_bars - 1) * inner_gap + group_gap)
        for j in range(n_bars):
            x_positions.append(base + j * (bar_width + inner_gap))
    x_positions = np.array(x_positions).reshape(n_constructs, n_bars)
    
    # Group centers for x-tick labels
    group_centers = x_positions.mean(axis=1)
    
    # Plot bars for each construct
    for construct_idx, construct in enumerate(CONSTRUCTS):
        construct_name = construct["short"]
        construct_data = data[construct_name]
        
        means = construct_data["means"]
        stds = construct_data["stds"]
        
        for bar_idx, (proportion, mean, std) in enumerate(zip(EXP1_PROPORTIONS, means, stds)):
            x_pos = x_positions[construct_idx, bar_idx]
            color = COLORS[proportion]
            
            # Plot bar
            ax.bar(x_pos, mean, bar_width, color=color, alpha=0.9, edgecolor='none', zorder=3)
            
            # Plot error bar
            if not np.isnan(std):
                ax.errorbar(x_pos, mean, yerr=std, fmt='none', ecolor='#2C2C2C',
                           elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=4, alpha=0.7)
    
    # Styling
    ax.set_ylabel("AUPRC (macro)", fontsize=11, color='#2C2C2C', weight='bold')
    ax.set_title("Experiment 1: Data-Portion Sweep", pad=10, fontsize=13, weight='bold', color='#1A1A1A')
    ax.set_xticks(group_centers)
    ax.set_xticklabels([c["short"] for c in CONSTRUCTS], fontsize=10, color='#2C2C2C')
    ax.set_ylim(0.30, 0.55)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.8, alpha=0.7, color='#BFC7D5', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#A9B2C3')
    ax.spines['bottom'].set_color('#A9B2C3')
    
    # Add background bands for each construct
    for i in range(n_constructs):
        ax.add_patch(Rectangle(
            (x_positions[i, 0] - bar_width / 2, 0.30),
            n_bars * bar_width + (n_bars - 1) * inner_gap,
            0.40,
            facecolor='#F4F6FA',
            edgecolor='none',
            zorder=0,
            alpha=0.6
        ))


def plot_experiment_2(ax, data: Dict):
    """
    Plot Experiment 2: Epoch sweep.
    5 lines (one per proportion) + 1 point (zero-shot).
    Each line shows average AUPRC across all constructs.
    """
    # Plot zero-shot point
    zero_shot_mean = data["zero_shot"]["mean"]
    zero_shot_std = data["zero_shot"]["std"]
    ax.scatter(0, zero_shot_mean, s=120, color=COLORS[0.0], marker='o', 
              edgecolor='white', linewidth=1.5, zorder=5, label="Zero-shot")
    
    # Plot lines for each proportion
    for proportion in EXP2_PROPORTIONS:
        proportion_data = data[proportion]
        epochs = proportion_data["epochs"]
        means = proportion_data["means"]
        
        color = COLORS[proportion]
        label = f"{int(proportion * 100)}%"
        
        # Plot line
        ax.plot(epochs, means, color=color, linewidth=2.5, marker='o', markersize=6,
               markeredgecolor='white', markeredgewidth=1.0, label=label, alpha=0.9, zorder=3)
    
    # Styling
    ax.set_ylabel("AUPRC (macro)\n(averaged over constructs)", fontsize=11, color='#2C2C2C', weight='bold')
    ax.set_xlabel("Epoch", fontsize=11, color='#2C2C2C', weight='bold')
    ax.set_title("Experiment 2: Epoch Sweep", pad=10, fontsize=13, weight='bold', color='#1A1A1A')
    ax.set_xticks([0] + EXP2_EPOCHS)
    ax.set_xticklabels(["0\n(Zero-shot)"] + [str(e) for e in EXP2_EPOCHS], fontsize=10, color='#2C2C2C')
    ax.set_ylim(0.35, 0.50)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.8, alpha=0.7, color='#BFC7D5', zorder=1)
    ax.xaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.5, color='#E0E0E0', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#A9B2C3')
    ax.spines['bottom'].set_color('#A9B2C3')


def plot_experiment_3(ax, data: Dict):
    """
    Plot Experiment 3: Frozen backbone with item split.
    5 clusters (one per construct), each with 2 bars (zero-shot vs frozen backbone).
    """
    n_constructs = len(CONSTRUCTS)
    n_bars = 2  # zero-shot and frozen backbone
    bar_width = 0.25
    inner_gap = 0.08
    group_gap = 0.8
    
    # Calculate x-positions for each bar
    x_positions = []
    for i in range(n_constructs):
        base = i * (n_bars * bar_width + (n_bars - 1) * inner_gap + group_gap)
        for j in range(n_bars):
            x_positions.append(base + j * (bar_width + inner_gap))
    x_positions = np.array(x_positions).reshape(n_constructs, n_bars)
    
    # Group centers for x-tick labels
    group_centers = x_positions.mean(axis=1)
    
    # Colors for zero-shot and frozen backbone
    bar_colors = [COLORS[0.0], "#E8684A"]  # Dark gray for zero-shot, red-orange for frozen backbone
    bar_labels = ["Zero-shot", "Frozen Backbone\n(80% train)"]
    
    # Plot bars for each construct
    for construct_idx, construct in enumerate(CONSTRUCTS):
        construct_name = construct["short"]
        construct_data = data[construct_name]
        
        # Zero-shot bar
        zs_mean, zs_std = construct_data["zero_shot"]
        x_pos = x_positions[construct_idx, 0]
        ax.bar(x_pos, zs_mean, bar_width, color=bar_colors[0], alpha=0.9, edgecolor='none', zorder=3)
        if not np.isnan(zs_std):
            ax.errorbar(x_pos, zs_mean, yerr=zs_std, fmt='none', ecolor='#2C2C2C',
                       elinewidth=1.0, capsize=3, capthick=1.0, zorder=4, alpha=0.7)
        
        # Frozen backbone bar
        fb_mean, fb_std = construct_data["frozen_backbone"]
        x_pos = x_positions[construct_idx, 1]
        ax.bar(x_pos, fb_mean, bar_width, color=bar_colors[1], alpha=0.9, edgecolor='none', zorder=3)
        if not np.isnan(fb_std):
            ax.errorbar(x_pos, fb_mean, yerr=fb_std, fmt='none', ecolor='#2C2C2C',
                       elinewidth=1.0, capsize=3, capthick=1.0, zorder=4, alpha=0.7)
    
    # Styling
    ax.set_ylabel("AUPRC (macro)", fontsize=11, color='#2C2C2C', weight='bold')
    ax.set_title("Experiment 3: Frozen Backbone (Item Split)", pad=10, fontsize=13, weight='bold', color='#1A1A1A')
    ax.set_xticks(group_centers)
    ax.set_xticklabels([c["short"] for c in CONSTRUCTS], fontsize=10, color='#2C2C2C')
    ax.set_ylim(0.30, 0.70)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.8, alpha=0.7, color='#BFC7D5', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#A9B2C3')
    ax.spines['bottom'].set_color('#A9B2C3')
    
    # Add background bands for each construct
    for i in range(n_constructs):
        ax.add_patch(Rectangle(
            (x_positions[i, 0] - bar_width / 2, 0.30),
            n_bars * bar_width + (n_bars - 1) * inner_gap,
            0.40,
            facecolor='#F4F6FA',
            edgecolor='none',
            zorder=0,
            alpha=0.6
        ))


def create_visualization(
    exp1_data: Dict,
    exp2_data: Dict,
    exp3_data: Dict,
    output_dir: str,
    figure_name: str,
    show: bool
):
    """
    Create the main visualization with 3 horizontal panels.
    """
    # Style settings
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
    
    # Create figure with 3 horizontal panels
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    fig.patch.set_facecolor('#FAFAFB')
    
    # Plot each experiment
    plot_experiment_1(axes[0], exp1_data)
    plot_experiment_2(axes[1], exp2_data)
    plot_experiment_3(axes[2], exp3_data)
    
    # Create custom legends
    # Legend for Experiment 1
    handles_exp1 = []
    labels_exp1 = []
    for proportion, label in zip(EXP1_PROPORTIONS, EXP1_LABELS):
        from matplotlib.patches import Patch
        handles_exp1.append(Patch(facecolor=COLORS[proportion], edgecolor='none'))
        labels_exp1.append(label)
    
    axes[0].legend(handles_exp1, labels_exp1, loc='upper left', frameon=False, 
                  fontsize=9, ncol=3, columnspacing=1.2)
    
    # Legend for Experiment 2
    axes[1].legend(loc='upper left', frameon=False, fontsize=9, ncol=2)
    
    # Legend for Experiment 3
    from matplotlib.patches import Patch
    handles_exp3 = [
        Patch(facecolor=COLORS[0.0], edgecolor='none'),
        Patch(facecolor="#E8684A", edgecolor='none')
    ]
    labels_exp3 = ["Zero-shot", "Frozen Backbone (80% train)"]
    axes[2].legend(handles_exp3, labels_exp3, loc='upper left', frameon=False, fontsize=9)
    
    plt.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.12, wspace=0.25)
    
    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{figure_name}.png")
    pdf_path = os.path.join(output_dir, f"{figure_name}.pdf")
    
    plt.savefig(png_path, bbox_inches="tight", facecolor='#FAFAFB')
    plt.savefig(pdf_path, bbox_inches="tight", facecolor='#FAFAFB')
    print(f"Saved figure to: {png_path}")
    print(f"Saved figure to: {pdf_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ----------------------------- Main -----------------------------

def main():
    args = parse_args()
    
    print("=" * 60)
    print("OOD Adaptation Experiment Visualization")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load data
    print("Loading data...")
    try:
        exp1_df = load_exp1_data(args.input_dir)
        exp2_df = load_exp2_data(args.input_dir)
        exp3_df = load_exp3_data(args.input_dir)
        print(f"  Experiment 1: {len(exp1_df)} rows")
        print(f"  Experiment 2: {len(exp2_df)} rows")
        print(f"  Experiment 3: {len(exp3_df)} rows")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required CSV files exist in the input directory.")
        return
    
    # Extract data for plotting
    print("\nExtracting data for plotting...")
    exp1_data = get_exp1_data_for_plotting(exp1_df)
    exp2_data = get_exp2_data_for_plotting(exp2_df, exp1_df)
    exp3_data = get_exp3_data_for_plotting(exp3_df, exp1_df)
    print("  Done!")
    
    # Create visualization
    print("\nCreating visualization...")
    create_visualization(
        exp1_data,
        exp2_data,
        exp3_data,
        args.output_dir,
        args.figure_name,
        show=not args.no_show
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

