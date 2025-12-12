"""
BSCX Ablation Results Visualization

This script visualizes stress test results from multichannel_fusion/runner.py.

Usage Examples:

All-constructs aggregated (all splits combined):
   python experiments/viz/BSCX_ablation_results.py \
       --csv experiments/multichannel_fusion_results/stress_test_results_all_constructs.csv \
       --aggregation-level all_constructs \
       --split-filter all \
       --outdir experiments/viz_results/bscx
"""

import os
import argparse
from typing import Optional, List, Dict

import numpy as npsss
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Global style 
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 350,
    "savefig.dpi": 350,
    "axes.linewidth": 1.2,
})

# Same palette as the reference code
PALETTE = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#FF6B6B"]

# Explicit overrides for key variants (case-insensitive)
COLOR_OVERRIDES = {
    "bscx": "#F6BD16",        # golden
    "concat_proj": "#61DDAA",  # green
    "concat-proj": "#61DDAA",  # green (with hyphen)
    "uniform_avg": "#5B8FF9", # blue
    "uniform-avg": "#5B8FF9", # blue (with hyphen)
    "proj_only": "#65789B",   # gray-blue
}

# Mapping from model variant names to display names for legend
VARIANT_DISPLAY_NAMES = {
    "bscx": "BSC-X",
    "concat_proj": "Concat + Proj",
    "concat-proj": "Concat + Proj",
    "uniform_avg": "Uniform Avg",
    "uniform-avg": "Uniform Avg",
    "proj_only": "Proj Only",
}


def get_display_name(variant_name: str) -> str:
    """Get display name for a model variant, with fallback to original name."""
    key = variant_name.lower().strip()
    return VARIANT_DISPLAY_NAMES.get(key, variant_name)


def compute_ylim(values, margin_low=0.03, margin_high=0.03):
    values = np.asarray(values, dtype=float)
    vmin, vmax = float(values.min()), float(values.max())
    
    # Calculate data range
    data_range = vmax - vmin
    
    # Use adaptive margins: smaller margins for larger ranges, but ensure minimum padding
    if data_range > 0:
        # Use 2-3% of data range as margin, but at least 0.01
        adaptive_margin = max(0.01, data_range * 0.02)
        margin_low = min(margin_low, adaptive_margin)
        margin_high = min(margin_high, adaptive_margin)
    
    lo = max(0.0, vmin - margin_low)
    hi = min(1.0, vmax + margin_high)
    
    # Use finer rounding (0.01 increments instead of 0.05) for better precision
    lo = np.floor(lo * 100) / 100.0
    hi = np.ceil(hi * 100) / 100.0
    
    # Ensure minimum range, but use a smaller minimum if data range is small
    min_range = max(0.10, data_range * 1.1)  # At least 10% more than data range
    if hi - lo < min_range:
        # Center the range around the data
        center = (vmin + vmax) / 2.0
        lo = max(0.0, center - min_range / 2.0)
        hi = min(1.0, center + min_range / 2.0)
        # Re-round
        lo = np.floor(lo * 100) / 100.0
        hi = np.ceil(hi * 100) / 100.0
    
    return lo, hi


def plot_corruption_trends(
    csv_path: str,
    output_dir: str,
    only_variants: Optional[List[str]] = None,
    metric_name: str = "auprc_macro",
    show: bool = False,
    fit_line: bool = False,
    aggregation_level: str = "per_fold",
    split_filter: Optional[str] = None,
    head_filter: Optional[str] = None,
):
    """
    Creates a single figure with up to 5 horizontally stacked subplots:
        Stream Dropout | Channel Dropout | Temporal Band | Jitter | Noise

    Each subplot:
      - X-axis: corruption_param (with specific tick sets per corruption type)
      - Y-axis: metric_name (default: auprc_macro)
      - One line per model_variant:
          * For per_fold: Transparent per-fold points + means
          * For aggregated: Mean ± std error bars or just means

    Args:
        csv_path: Path to CSV file (per_fold, per_construct, or all_constructs)
        aggregation_level: "per_fold", "per_construct", or "all_constructs"
        split_filter: Filter by split (e.g., "individual", "group", "all", or None for all)
        head_filter: Filter by head (construct name, or None for all)
    """

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Filter by metric
    df = df[df["metric"] == metric_name].copy()
    if df.empty:
        print(f"No rows with metric == '{metric_name}'.")
        return

    # Optional: filter variants
    if only_variants is not None:
        only_variants = [v.strip() for v in only_variants if v.strip()]
        df = df[df["model_variant"].isin(only_variants)].copy()
        if df.empty:
            print("No rows left after filtering by model_variant.")
            return

    # Handle head filtering based on aggregation level
    if aggregation_level == "all_constructs":
        # For all_constructs, head should be "all_constructs"
        # If head_filter is None, only show all_constructs rows
        # If head_filter is specified, use it (though it should typically be "all_constructs")
        if head_filter is None:
            df = df[df["head"] == "all_constructs"].copy()
        else:
            df = df[df["head"] == head_filter].copy()
    elif aggregation_level == "per_construct":
        # For per_construct, head should NOT be "all_constructs"
        # If head_filter is None, exclude all_constructs rows
        # If head_filter is specified, use it
        if head_filter is None:
            df = df[df["head"] != "all_constructs"].copy()
        else:
            df = df[df["head"] == head_filter].copy()
    else:  # per_fold
        # For per_fold, use head_filter as-is
        if head_filter is not None:
            df = df[df["head"] == head_filter].copy()
    
    if df.empty:
        print(f"No rows left after filtering by head (aggregation_level={aggregation_level}).")
        return

    # Filter by split if specified
    if split_filter is not None:
        df = df[df["split"] == split_filter].copy()
        if df.empty:
            print(f"No rows left after filtering by split='{split_filter}'.")
            return

    # Numeric corruption_param
    df["corruption_param"] = pd.to_numeric(df["corruption_param"], errors="coerce")
    df = df.dropna(subset=["corruption_param"])

    # Determine if we have aggregated data (mean column) or per-fold data (value column)
    has_mean = "mean" in df.columns
    has_value = "value" in df.columns
    
    if has_mean:
        # Use mean from aggregated data
        df["plot_value"] = pd.to_numeric(df["mean"], errors="coerce")
        has_std = "std" in df.columns
        if has_std:
            df["plot_std"] = pd.to_numeric(df["std"], errors="coerce")
    elif has_value:
        # Use value from per-fold data
        df["plot_value"] = pd.to_numeric(df["value"], errors="coerce")
        has_std = False
    else:
        print("CSV must contain either 'mean' (aggregated) or 'value' (per-fold) column.")
        return

    df = df.dropna(subset=["plot_value"])

    if df.empty:
        print("No valid rows after cleaning corruption_param/plot_value.")
        return

    # Normalize corruption_type for matching
    df["corr_norm"] = (
        df["corruption_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", " ")
        .str.replace("_", " ")
    )

    # Map original corr_type -> corr_norm
    corr_norm_map = {}
    for orig, norm in zip(df["corruption_type"], df["corr_norm"]):
        corr_norm_map[orig] = norm

    # Determine which corruption_type maps to which logical panel
    # BSCX corruption types: temporal_band, jitter, noise
    panel_kinds = ["jitter", "noise", "temporal_band"]
    ordered_corr_info = []  # list of (kind, orig_corr_type)

    for kind in panel_kinds:
        found_type = None
        for orig, norm in corr_norm_map.items():
            if kind == "temporal_band" and ("temporal" in norm or "band" in norm):
                found_type = orig
                break
            elif kind == "jitter" and "jitter" in norm:
                found_type = orig
                break
            elif kind == "noise" and "noise" in norm and "dropout" not in norm:
                found_type = orig
                break
        if found_type is not None:
            ordered_corr_info.append((kind, found_type))

    if not ordered_corr_info:
        print("No recognized corruption_types (temporal_band/jitter/noise) found in CSV.")
        return

    n_panels = len(ordered_corr_info)

    # Figure + layout (no shared y-range: each subplot gets its own)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(13, 3.6),
        sharey=False,
    )
    if n_panels == 1:
        axes = [axes]

    fig.patch.set_facecolor("#FAFAFB")  # outer figure background

    # Consistent colors across variants with overrides
    all_variants = sorted(df["model_variant"].unique().tolist())
    color_map: Dict[str, str] = {}
    palette_index = 0
    for mv in all_variants:
        key = mv.lower()
        if key in COLOR_OVERRIDES:
            color_map[mv] = COLOR_OVERRIDES[key]
        else:
            color_map[mv] = PALETTE[palette_index % len(PALETTE)]
            palette_index += 1

    # For collecting legend handles only once
    handles_for_legend = []
    labels_for_legend = []

    # ------------------------------------------------------------------
    # Plot each corruption type into its panel
    # ------------------------------------------------------------------
    for ax, (kind, corr_type) in zip(axes, ordered_corr_info):
        df_corr = df[df["corruption_type"] == corr_type].copy()
        if df_corr.empty:
            continue

        # Panel background like the bar plots
        ax.set_facecolor("#F4F6FA")
        ax.set_axisbelow(True)

        for mv, df_mv in df_corr.groupby("model_variant"):
            col = color_map[mv]

            # Group by corruption_param and compute means
            grouped = df_mv.groupby("corruption_param")
            xs_means = []
            ys_means = []
            ys_stds = []

            for cp, df_cp in grouped:
                xs_means.append(float(cp))
                ys_means.append(float(df_cp["plot_value"].mean()))
                if has_std and "plot_std" in df_cp.columns:
                    ys_stds.append(float(df_cp["plot_std"].mean()))
                else:
                    ys_stds.append(0.0)

            xs_means = np.asarray(xs_means, dtype=float)
            ys_means = np.asarray(ys_means, dtype=float)
            ys_stds = np.asarray(ys_stds, dtype=float)

            if len(xs_means) == 0:
                continue

            # Sort by corruption_param
            order = np.argsort(xs_means)
            xs_means = xs_means[order]
            ys_means = ys_means[order]
            ys_stds = ys_stds[order]

            # Get display name for this variant
            display_name = get_display_name(mv)

            # For per-fold data, also show individual fold points
            if has_value and "fold" in df_mv.columns:
                # Show per-fold scatter points
                for cp in sorted(df_mv["corruption_param"].unique()):
                    df_cp = df_mv[df_mv["corruption_param"] == cp]
                    fold_vals = df_cp["plot_value"].values
                    ax.scatter(
                        [cp] * len(fold_vals),
                        fold_vals,
                        color=col,
                        alpha=0.25,
                        s=30,
                        edgecolors="none",
                        zorder=2,
                    )

            if fit_line:
                # Polynomial fit (no solid mean circles)
                if len(xs_means) >= 2:
                    deg = min(2, len(xs_means) - 1)
                    coeffs = np.polyfit(xs_means, ys_means, deg=deg)
                    poly = np.poly1d(coeffs)

                    x_smooth = np.linspace(xs_means.min(), xs_means.max(), 200)
                    y_smooth = poly(x_smooth)

                    line = ax.plot(
                        x_smooth,
                        y_smooth,
                        color=col,
                        linestyle="-",
                        linewidth=2.0,
                        label=display_name,
                        zorder=3,
                    )[0]
                else:
                    # Single point: just show its mean as a marker (no line)
                    line = ax.scatter(
                        xs_means,
                        ys_means,
                        color=col,
                        alpha=0.95,
                        s=55,
                        edgecolors="white",
                        linewidths=0.6,
                        label=display_name,
                        zorder=4,
                    )
            else:
                # Straight line connection between mean points + solid mean circles
                line = ax.plot(
                    xs_means,
                    ys_means,
                    color=col,
                    linestyle="-",
                    linewidth=2.0,
                    label=display_name,
                    zorder=3,
                )[0]
                ax.scatter(
                    xs_means,
                    ys_means,
                    color=col,
                    alpha=0.98,
                    s=55,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=4,
                )
                # Optionally show error bars if we have std
                if has_std and np.any(ys_stds > 0):
                    ax.errorbar(
                        xs_means,
                        ys_means,
                        yerr=ys_stds,
                        color=col,
                        alpha=0.4,
                        linewidth=0.8,
                        capsize=3,
                        capthick=0.8,
                        zorder=1,
                    )

            # Collect handle for legend (one per variant)
            if display_name not in labels_for_legend:
                labels_for_legend.append(display_name)
                handles_for_legend.append(line)

        # Titles and x-axis labels per panel
        if kind == "temporal_band":
            ax.set_title("Temporal Band Masking", pad=6, color="#1A1A1A")
            ax.set_xlabel("Masked Sequence Fraction")
            desired_ticks = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
        elif kind == "jitter":
            ax.set_title("Misalignment Jitter", pad=6, color="#1A1A1A")
            ax.set_xlabel("Max Random Temporal Jitter (± time steps)")
            desired_ticks = [0, 5, 10, 15, 20, 25]
        else:  # "noise"
            ax.set_title("Feature Noise", pad=6, color="#1A1A1A")
            ax.set_xlabel("Relative Gaussian Noise Scale (σ_noise / σ_feature)")
            desired_ticks = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]

        # Keep only ticks where we actually have data
        cp_vals = np.unique(df_corr["corruption_param"].values.astype(float))
        valid_ticks = [t for t in desired_ticks if np.any(np.isclose(cp_vals, t, atol=1e-8))]

        ax.set_xticks(valid_ticks)
        # Format tick labels: two decimals for most, but for jitter use integers
        if kind == "jitter":
            ax.set_xticklabels([f"{int(t)}" for t in valid_ticks])
        else:
            ax.set_xticklabels([f"{t:.2f}" for t in valid_ticks])

        # Grid & spines to match style
        ax.yaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.7, color="#BFC7D5")
        ax.xaxis.grid(False)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#A9B2C3")

    # Shared Y label
    ylabel = f"{metric_name.upper().replace('_', ' ').replace('MACRO', '')} (macro)"
    if split_filter:
        if split_filter == "all":
            ylabel += f"\n(averaged over constructs)"
        else:
            ylabel += f" ({split_filter.capitalize()})"
    if head_filter:
        ylabel += f" ({head_filter})"

    if split_filter == "all":
        x = 0.015
    else:
        x = 0.02

    fig.text(
        x,
        0.5,
        ylabel,
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12,
    )

    # Shared legend (top center), no subtitle
    if handles_for_legend and labels_for_legend:
        fig.legend(
            handles_for_legend,
            labels_for_legend,
            loc="upper center",
            ncol=min(len(labels_for_legend), 5),
            frameon=False,
            bbox_to_anchor=(0.5, 1.03),
            columnspacing=2.5,
            handlelength=1.8,
            prop={'weight': 'bold', 'size': 12},
        )

    # Layout
    plt.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.22, wspace=0.20)

    suffix = "poly" if fit_line else "linear"
    agg_suffix = aggregation_level
    if split_filter:
        agg_suffix += f"_{split_filter}"
    if head_filter:
        agg_suffix += f"_{head_filter}"
    
    out_path_png = os.path.join(
        output_dir, f"corruption_trends__{metric_name}__{agg_suffix}__{suffix}.png"
    )
    out_path_pdf = os.path.join(
        output_dir, f"corruption_trends__{metric_name}__{agg_suffix}__{suffix}.pdf"
    )
    fig.savefig(out_path_png, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    print(f"Saved: {out_path_png}")
    print(f"Saved: {out_path_pdf}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot corruption sweeps with per-fold points and either "
            "quadratic trend lines or straight-line connections per model variant.\n"
            "Produces a single figure with up to 5 horizontal subplots "
            "(Stream Dropout, Channel Dropout, Temporal Band Mask, Jitter, Noise).\n"
            "Default metric: auprc_macro (AUPRC)."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file (with columns including metric,value/mean,fold,corruption_type).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots_corruptions",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Comma-separated list of model_variant names to include "
            "(e.g. 'bscx,concat_proj,uniform_avg'). "
            "If omitted, all variants present in the CSV are used."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="auprc_macro",
        help="Metric name to plot (default: auprc_macro).",
    )
    parser.add_argument(
        "--fit-line",
        action="store_true",
        help=(
            "If set, fit a 2nd-order polynomial to the mean points per variant "
            "and plot the smooth curve (no solid mean circles). "
            "If not set, connect mean points with straight lines and show solid circles."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, display plots interactively.",
    )
    parser.add_argument(
        "--aggregation-level",
        type=str,
        default="per_fold",
        choices=["per_fold", "per_construct", "all_constructs"],
        help=(
            "Aggregation level: 'per_fold' uses per_fold CSV, "
            "'per_construct' uses per_construct CSV (aggregated over folds, per construct), "
            "'all_constructs' uses all_constructs CSV (aggregated across all constructs)."
        ),
    )
    parser.add_argument(
        "--split-filter",
        type=str,
        default=None,
        help=(
            "Filter by split type (e.g., 'individual', 'group', 'all'). "
            "For all_constructs aggregation level, 'all' aggregates across both individual and group splits. "
            "If None, shows all splits."
        ),
    )
    parser.add_argument(
        "--head-filter",
        type=str,
        default=None,
        help=(
            "Filter by specific head/construct name (e.g., 'Engagement', 'Synchrony'). "
            "If None, shows all heads."
        ),
    )

    args = parser.parse_args()

    if args.variants is not None:
        variants = [v.strip() for v in args.variants.split(",")]
    else:
        variants = None

    plot_corruption_trends(
        csv_path=args.csv,
        output_dir=args.outdir,
        only_variants=variants,
        metric_name=args.metric,
        show=args.show,
        fit_line=args.fit_line,
        aggregation_level=args.aggregation_level,
        split_filter=args.split_filter,
        head_filter=args.head_filter,
    )


if __name__ == "__main__":
    main()

