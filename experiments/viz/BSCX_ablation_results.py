import os
import argparse
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Global style to match the barplot aesthetic
# ------------------------------------------------------------------
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

# Same palette as the reference code
PALETTE = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16"]

# Explicit overrides for key variants (case-insensitive)
COLOR_OVERRIDES = {
    "bscx": "#F6BD16",        # golden
    "concat_proj": "#61DDAA", # green
    "uniform_avg": "#5B8FF9", # blue
}

# Mapping from model variant names to display names for legend
VARIANT_DISPLAY_NAMES = {
    "bscx": "BSC-X",
    "concat_proj": "Concat + MLP",
    "uniform_avg": "Mean Pooling",
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
):
    """
    Creates a single figure with up to 3 horizontally stacked subplots:
        Misalignment Jitter | Feature Noise | Temporal Band Masking

    Each subplot:
      - X-axis: corruption_param (with specific tick sets per corruption type)
      - Y-axis: AUPRC
      - One line per model_variant:
          * Transparent per-fold points
          * Means per corruption_param connected by lines or smoothed poly fit.
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

    # Numeric corruption_param
    df["corruption_param"] = pd.to_numeric(df["corruption_param"], errors="coerce")
    df = df.dropna(subset=["corruption_param"])

    # Numeric metric value
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    if df.empty:
        print("No valid rows after cleaning corruption_param/value.")
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

    # Determine which corruption_type maps to which logical panel (jitter/noise/temporal)
    panel_kinds = ["jitter", "noise", "temporal"]
    ordered_corr_info = []  # list of (kind, orig_corr_type)

    for kind in panel_kinds:
        found_type = None
        for orig, norm in corr_norm_map.items():
            if kind == "jitter" and "jitter" in norm:
                found_type = orig
                break
            elif kind == "noise" and "noise" in norm:
                found_type = orig
                break
            elif kind == "temporal" and "temporal" in norm:
                found_type = orig
                break
        if found_type is not None:
            ordered_corr_info.append((kind, found_type))

    if not ordered_corr_info:
        print("No recognized corruption_types (jitter/noise/temporal) found in CSV.")
        return

    n_panels = len(ordered_corr_info)

    # Figure + layout (no shared y-range: each subplot gets its own)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(16, 3.6),
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

        # Y-limits independently based on data in this panel
        # y_lo, y_hi = compute_ylim(df_corr["value"].values)
        # ax.set_ylim(y_lo, y_hi)

        for mv, df_mv in df_corr.groupby("model_variant"):
            col = color_map[mv]

            if "fold" not in df_mv.columns:
                raise ValueError("CSV must contain a 'fold' column for per-fold plotting.")

            # 1) Compute per-fold means for each corruption_param
            cp_fold_means: Dict[float, List[float]] = {}
            grouped = df_mv.groupby(["corruption_param", "fold"])
            for (cp, fold), df_cp_fold in grouped:
                v_mean = df_cp_fold["value"].mean()
                cp_fold_means.setdefault(cp, []).append(float(v_mean))

            if not cp_fold_means:
                continue

            xs_means = []
            ys_means = []

            # 2) Scatter per-fold points and compute mean per corruption_param
            for cp in sorted(cp_fold_means.keys()):
                fold_vals = np.asarray(cp_fold_means[cp], dtype=float)

                # Transparent per-fold circles
                ax.scatter(
                    [cp] * len(fold_vals),
                    fold_vals,
                    color=col,
                    alpha=0.25,
                    s=30,
                    edgecolors="none",
                    zorder=2,
                )

                mean_cp = float(fold_vals.mean())
                xs_means.append(cp)
                ys_means.append(mean_cp)

            xs_means = np.asarray(xs_means, dtype=float)
            ys_means = np.asarray(ys_means, dtype=float)

            if len(xs_means) == 0:
                continue

            # Sort by corruption_param
            order = np.argsort(xs_means)
            xs_means = xs_means[order]
            ys_means = ys_means[order]

            # Get display name for this variant
            display_name = get_display_name(mv)

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

            # Collect handle for legend (one per variant)
            if display_name not in labels_for_legend:
                labels_for_legend.append(display_name)
                handles_for_legend.append(line)

        # Titles and x-axis labels per panel
        if kind == "jitter":
            ax.set_title("Misalignment Jitter", pad=6, color="#1A1A1A")
            ax.set_xlabel("Max Random Temporal Jitter (± time steps)")
            desired_ticks = [0, 5, 10, 15, 20, 25]
        elif kind == "noise":
            ax.set_title("Feature Noise", pad=6, color="#1A1A1A")
            ax.set_xlabel("Relative Gaussian Noise Scale (σ_noise / σ_feature)")
            desired_ticks = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
        else:  # "temporal"
            ax.set_title("Temporal Band Masking", pad=6, color="#1A1A1A")
            ax.set_xlabel("Masked Sequence Fraction")
            desired_ticks = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]

        # Keep only ticks where we actually have data
        cp_vals = np.unique(df_corr["corruption_param"].values.astype(float))
        valid_ticks = [t for t in desired_ticks if np.any(np.isclose(cp_vals, t, atol=1e-8))]

        ax.set_xticks(valid_ticks)
        # Format tick labels: ints for jitter, two decimals for others
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

    # Shared Y label (left side of figure): AUPRC
    fig.text(
        0.02,
        0.5,
        "Mean AUPRC over all Constructs",
        va="center",
        rotation="vertical",
        fontsize=11,
    )

    # Shared legend (top center), no subtitle
    if handles_for_legend and labels_for_legend:
        fig.legend(
            handles_for_legend,
            labels_for_legend,
            loc="upper center",
            ncol=min(len(labels_for_legend), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 1.03),
            columnspacing=2.5,
            handlelength=1.8,
            fontsize=9,
        )

    # Layout
    plt.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.22, wspace=0.20)

    suffix = "poly" if fit_line else "linear"
    out_path_png = os.path.join(
        output_dir, f"corruption_trends__{metric_name}__{suffix}.png"
    )
    out_path_pdf = os.path.join(
        output_dir, f"corruption_trends__{metric_name}__{suffix}.pdf"
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
            "Produces a single figure with up to 3 horizontal subplots "
            "(Misalignment Jitter, Feature Noise, Temporal Band Masking)."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the per-run CSV file (with columns including metric,value,fold,corruption_type).",
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
            "(e.g. 'bscx,concar_proj,uniform_avg'). "
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
    )


if __name__ == "__main__":
    main()









# import os
# import argparse
# from typing import Optional, List, Dict

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# def plot_corruption_trends(
#     csv_path: str,
#     output_dir: str,
#     only_variants: Optional[List[str]] = None,
#     metric_name: str = "auprc_macro",
#     show: bool = False,
#     fit_line: bool = False,
# ):
#     """
#     Reads per-run CSV (with 'value' column) and, for each corruption_type, creates ONE plot:

#       - X-axis: corruption_param
#       - Y-axis: metric_name (default: auprc_macro)

#     For each corruption_type and model_variant:
#       - For each corruption_param, compute per-fold means:
#           value_mean(fold) = mean over all rows matching that fold, head, split, etc.
#       - Always plot per-fold means as transparent circles.
#       - If fit_line == True:
#           * Fit a 2nd-order polynomial on (corruption_param, mean_across_folds).
#           * Plot the polynomial curve.
#           * Do NOT draw solid mean circles.
#       - If fit_line == False:
#           * Plot solid mean circles at each corruption_param.
#           * Connect the mean points with straight lines.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     df = pd.read_csv(csv_path)

#     # Filter by metric
#     df = df[df["metric"] == metric_name].copy()
#     if df.empty:
#         print(f"No rows with metric == '{metric_name}'.")
#         return

#     # Optional: filter variants
#     if only_variants is not None:
#         only_variants = [v.strip() for v in only_variants if v.strip()]
#         df = df[df["model_variant"].isin(only_variants)].copy()
#         if df.empty:
#             print("No rows left after filtering by model_variant.")
#             return

#     # Numeric corruption_param
#     df["corruption_param"] = pd.to_numeric(df["corruption_param"], errors="coerce")
#     df = df.dropna(subset=["corruption_param"])

#     # Numeric metric value
#     df["value"] = pd.to_numeric(df["value"], errors="coerce")
#     df = df.dropna(subset=["value"])

#     # Consistent colors across variants
#     all_variants = sorted(df["model_variant"].unique().tolist())
#     color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     color_map: Dict[str, str] = {
#         mv: color_cycle[i % len(color_cycle)] for i, mv in enumerate(all_variants)
#     }

#     # One plot per corruption_type
#     for corr_type, df_corr in df.groupby("corruption_type"):
#         fig, ax = plt.subplots(figsize=(8, 4))

#         for mv, df_mv in df_corr.groupby("model_variant"):
#             col = color_map[mv]

#             # 1) Compute per-fold means for each corruption_param
#             if "fold" not in df_mv.columns:
#                 raise ValueError("CSV must contain a 'fold' column for per-fold plotting.")

#             cp_fold_means: Dict[float, List[float]] = {}
#             grouped = df_mv.groupby(["corruption_param", "fold"])
#             for (cp, fold), df_cp_fold in grouped:
#                 v_mean = df_cp_fold["value"].mean()
#                 cp_fold_means.setdefault(cp, []).append(float(v_mean))

#             if not cp_fold_means:
#                 continue

#             xs_means = []
#             ys_means = []

#             # 2) Scatter per-fold points and compute mean per corruption_param
#             for cp in sorted(cp_fold_means.keys()):
#                 fold_vals = np.asarray(cp_fold_means[cp], dtype=float)

#                 # Transparent per-fold circles
#                 ax.scatter(
#                     [cp] * len(fold_vals),
#                     fold_vals,
#                     color=col,
#                     alpha=0.25,
#                     s=25,
#                     edgecolors="none",
#                 )

#                 mean_cp = float(fold_vals.mean())
#                 xs_means.append(cp)
#                 ys_means.append(mean_cp)

#             xs_means = np.asarray(xs_means, dtype=float)
#             ys_means = np.asarray(ys_means, dtype=float)

#             # Sort by corruption_param
#             order = np.argsort(xs_means)
#             xs_means = xs_means[order]
#             ys_means = ys_means[order]

#             if len(xs_means) == 0:
#                 continue

#             if fit_line:
#                 # 3a) Polynomial fit (no solid mean circles)
#                 if len(xs_means) >= 2:
#                     deg = min(2, len(xs_means) - 1)
#                     coeffs = np.polyfit(xs_means, ys_means, deg=deg)
#                     poly = np.poly1d(coeffs)

#                     x_smooth = np.linspace(xs_means.min(), xs_means.max(), 200)
#                     y_smooth = poly(x_smooth)

#                     ax.plot(
#                         x_smooth,
#                         y_smooth,
#                         color=col,
#                         linestyle="-",
#                         linewidth=2.0,
#                         label=mv,
#                     )
#                 else:
#                     # Single point: just show its mean as a marker (no line)
#                     ax.scatter(
#                         xs_means,
#                         ys_means,
#                         color=col,
#                         alpha=0.95,
#                         s=55,
#                         edgecolors="k",
#                         linewidths=0.5,
#                         label=mv,
#                         zorder=3,
#                     )
#             else:
#                 # 3b) Straight line connection between mean points + solid mean circles
#                 ax.plot(
#                     xs_means,
#                     ys_means,
#                     color=col,
#                     linestyle="-",
#                     linewidth=2.0,
#                     label=mv,
#                 )
#                 # Solid mean circles
#                 ax.scatter(
#                     xs_means,
#                     ys_means,
#                     color=col,
#                     alpha=0.95,
#                     s=55,
#                     edgecolors="k",
#                     linewidths=0.5,
#                     zorder=3,
#                 )

#         ax.set_xlabel("Corruption parameter")
#         ax.set_ylabel(f"{metric_name} (per-fold values; summary = means)")
#         ax.set_title(
#             f"{corr_type} – {metric_name} vs. corruption "
#             + ("(poly fit)" if fit_line else "(linear connect)")
#         )
#         ax.grid(True, alpha=0.3)
#         ax.legend(title="Model variant", fontsize=8)

#         safe_corr = str(corr_type).replace("/", "_")
#         suffix = "poly" if fit_line else "linear"
#         out_path = os.path.join(
#             output_dir, f"{safe_corr}__{metric_name}__{suffix}_all_variants.png"
#         )
#         fig.tight_layout()
#         fig.savefig(out_path, dpi=220)
#         print(f"Saved: {out_path}")

#         if show:
#             plt.show()
#         else:
#             plt.close(fig)


# def main():
#     parser = argparse.ArgumentParser(
#         description=(
#             "Plot corruption sweeps with per-fold points and either "
#             "quadratic trend lines or straight-line connections per model variant.\n"
#             "Uses per-run CSV (with 'value' column). One plot per corruption_type."
#         )
#     )
#     parser.add_argument(
#         "--csv",
#         type=str,
#         required=True,
#         help="Path to the per-run CSV file (with columns including metric,value,fold).",
#     )
#     parser.add_argument(
#         "--outdir",
#         type=str,
#         default="plots_corruptions",
#         help="Directory where plots will be saved.",
#     )
#     parser.add_argument(
#         "--variants",
#         type=str,
#         default=None,
#         help=(
#             "Comma-separated list of model_variant names to include "
#             "(e.g. 'bscx,bsxprojonly,concatprojfusion,uniformavgfusion'). "
#             "If omitted, all variants present in the CSV are used."
#         ),
#     )
#     parser.add_argument(
#         "--metric",
#         type=str,
#         default="auprc_macro",
#         help="Metric name to plot (default: auprc_macro).",
#     )
#     parser.add_argument(
#         "--fit-line",
#         action="store_true",
#         help=(
#             "If set, fit a 2nd-order polynomial to the mean points per variant "
#             "and plot the smooth curve (no solid mean circles). "
#             "If not set, connect mean points with straight lines and show solid circles."
#         ),
#     )
#     parser.add_argument(
#         "--show",
#         action="store_true",
#         help="If set, display plots interactively.",
#     )

#     args = parser.parse_args()

#     if args.variants is not None:
#         variants = [v.strip() for v in args.variants.split(",")]
#     else:
#         variants = None

#     plot_corruption_trends(
#         csv_path=args.csv,
#         output_dir=args.outdir,
#         only_variants=variants,
#         metric_name=args.metric,
#         show=args.show,
#         fit_line=args.fit_line,
#     )


# if __name__ == "__main__":
#     main()


