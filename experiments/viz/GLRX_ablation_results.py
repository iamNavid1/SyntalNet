"""
Reliability-Switch Multimodal Fusion Results Visualization

This script visualizes stress-test results from
`experiments/multimodal_reliability`, using the same aesthetics and
aggregation logic as `GLRX_ablation_results.py`.

- Three panels: Modality Dropout | Modality Noise | Modality Shuffle
- X-axis: corruption_param
- Y-axis: F1 (macro), aggregated over constructs and shown per corruption_type
- One line per fusion variant (GLR-X, UniformAvg, ConcatMLP, ...)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# ------------------------------------------------------------------
# Global style  (copied from GLRX_ablation_results.py)
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 350,
    "savefig.dpi": 350,
    "axes.linewidth": 1.2,
})

PALETTE = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#FF6B6B"]

COLOR_OVERRIDES = {
    "glrx": "#F6BD16",        # golden
    "glr_x": "#F6BD16",       # golden
    "concat_mlp": "#61DDAA",  # green
    "uniform_avg": "#5B8FF9", # blue
}

VARIANT_DISPLAY_NAMES = {
    "glrx": "GLR-X",
    "glr_x": "GLR-X",
    "uniform_avg": "Mean Pooling",
    "concat_mlp": "Concat + MLP",
}


def get_display_name(variant_name: str) -> str:
    key = variant_name.lower().strip()
    return VARIANT_DISPLAY_NAMES.get(key, variant_name)


def format_metric_name(metric_name: str) -> str:
    """Format metric name for display."""
    metric_lower = metric_name.lower().strip()
    metric_map = {
        "accuracy": "Accuracy",
        "f1_macro": "F1 Score",
        "f1_micro": "F1 Score",
        "f1": "F1 Score",
        "auprc": "AUPRC",
        "auroc": "AUROC",
    }
    return metric_map.get(metric_lower, metric_name.replace("_", " ").title())


# ------------------------------------------------------------------
# Data loading / aggregation
# ------------------------------------------------------------------

def collect_stress_results(
    root_dir: str,
    variants: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load and concatenate per-fold stress_tests.csv for all variants.

    Expected layout (variant-first):
      root_dir/
        <variant>/
          <cv_mode>/          e.g. kfold, logo
            fold_00/results/stress_tests.csv
            fold_01/results/stress_tests.csv
            ...

    Also supported: passing a single variant directory as root_dir:
      root_dir/
        <cv_mode>/
          fold_00/results/stress_tests.csv
          ...
    """
    root = Path(root_dir).resolve()

    records: List[pd.DataFrame] = []

    def _read_one(csv_path: Path, variant_name: str, fold_name: str, cv_mode: Optional[str]) -> None:
        df = pd.read_csv(csv_path)
        df["variant"] = variant_name
        df["fold"] = fold_name
        # Keep cv_mode around for debugging / filtering if needed downstream (harmless extra column).
        if cv_mode is not None:
            df["cv_mode"] = cv_mode
        records.append(df)

    def _iter_folds_under(cv_dir: Path, variant_name: str, cv_mode: Optional[str]) -> None:
        if not cv_dir.is_dir():
            return
        for fold_dir in cv_dir.iterdir():
            if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                continue
            csv_path = fold_dir / "results" / "stress_tests.csv"
            if csv_path.is_file():
                _read_one(csv_path, variant_name=variant_name, fold_name=fold_dir.name, cv_mode=cv_mode)

    cv_names = {"kfold", "logo"}

    root_subdirs = [p for p in root.iterdir() if p.is_dir()]

    # Case A: root is results root containing variants: root/<variant>/<cv_mode>/...
    root_looks_like_variant_root = any(
        any((sd / cv).is_dir() for cv in cv_names)
        for sd in root_subdirs
    )
    if root_looks_like_variant_root:
        for variant_dir in root_subdirs:
            variant_name = variant_dir.name
            if variants is not None and variant_name not in variants:
                continue
            for cv_dir in variant_dir.iterdir():
                if not cv_dir.is_dir():
                    continue
                _iter_folds_under(cv_dir, variant_name=variant_name, cv_mode=cv_dir.name)
    else:
        # Case B: root points at a single variant directory: root/<cv_mode>/...
        root_has_cv_dirs = any(p.is_dir() and p.name in cv_names for p in root_subdirs)
        if root_has_cv_dirs:
            if variants is not None and root.name not in variants:
                raise FileNotFoundError(
                    f"root-dir appears to be a single-variant folder ('{root.name}'), "
                    f"but it is not in --variants={variants}."
                )
            for cv_dir in root_subdirs:
                if not cv_dir.is_dir():
                    continue
                _iter_folds_under(cv_dir, variant_name=root.name, cv_mode=cv_dir.name)

    if not records:
        raise FileNotFoundError(
            "No stress_tests.csv found under root-dir. Expected one of:\n"
            f"  - {root_dir}/<variant>/<cv_mode>/fold_*/results/stress_tests.csv\n"
            f"  - {root_dir}/<cv_mode>/fold_*/results/stress_tests.csv  (if root-dir is a single variant)\n"
        )

    all_df = pd.concat(records, ignore_index=True)
    return all_df


def build_per_fold_f1_df(
    df: pd.DataFrame,
    split_filter: Optional[str] = "group",
    modality_filter: Optional[str] = "random_single",
    metric_name: str = "f1_macro",
) -> pd.DataFrame:
    """
    Construct a per-fold DataFrame with the same schema expected by
    GLRX_ablation_results.plot_corruption_trends:

      - model_variant
      - metric
      - value
      - fold
      - corruption_type
      - corruption_param

    Here, `value` is F1-macro averaged across all heads (constructs) for a given
    (variant, fold, corruption_type, corruption_param).
    """
    df = df.copy()

    # split_filter: "individual", "group", or "all"/None
    if split_filter is not None and split_filter != "all":
        df = df[df["split"] == split_filter]
    if modality_filter is not None:
        df = df[df["modality"] == modality_filter]

    if df.empty:
        raise ValueError("No rows left after filtering by split/modality.")

    df["corruption_param"] = pd.to_numeric(df["corruption_param"], errors="coerce")
    df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
    df = df.dropna(subset=["corruption_param", metric_name])

    # Aggregate over heads (constructs) within each fold
    grp = (
        df.groupby(
            ["variant", "fold", "corruption_type", "corruption_param"],
            as_index=False,
        )[metric_name]
        .mean()
        .rename(columns={metric_name: "value"})
    )

    # Rename columns to match GLRX plotting expectations
    grp = grp.rename(columns={"variant": "model_variant"})
    grp["metric"] = metric_name

    # Remember which split regime this aggregation corresponds to so that
    # plotting code can generate an informative y-axis label.
    # For split_filter == "all", we explicitly tag as "all".
    split_label = split_filter if split_filter is not None else "all"
    grp["split"] = split_label

    # Per-fold style; no precomputed mean/std columns here
    return grp[
        [
            "model_variant",
            "metric",
            "value",
            "fold",
            "corruption_type",
            "corruption_param",
            "split",
        ]
    ]


# ------------------------------------------------------------------
# Plotting (adapted from GLRX_ablation_results.plot_corruption_trends)
# ------------------------------------------------------------------

def plot_corruption_trends(
    df: pd.DataFrame,
    output_dir: str,
    metric_name: str = "f1_macro",
    show: bool = False,
) -> None:
    """
    Match GLRX_ablation_results: 3 horizontally stacked subplots
      Modality Dropout | Modality Noise | Modality Shuffle

    Per-fold visualization style:
      - Transparent per-fold points
      - Mean line across folds, with optional error bars
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter by metric
    df = df[df["metric"] == metric_name].copy()
    if df.empty:
        print(f"No rows with metric == '{metric_name}'.")
        return

    # Numeric corruption_param
    df["corruption_param"] = pd.to_numeric(df["corruption_param"], errors="coerce")
    df = df.dropna(subset=["corruption_param"])

    # Determine that we have per-fold data (value column)
    if "value" not in df.columns:
        print("Input must contain 'value' column for per-fold results.")
        return

    df["plot_value"] = pd.to_numeric(df["value"], errors="coerce")
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
    panel_kinds = ["dropout", "noise", "shuffle"]
    ordered_corr_info = []  # list of (kind, orig_corr_type)

    for kind in panel_kinds:
        found_type = None
        for orig, norm in corr_norm_map.items():
            if kind == "dropout" and "dropout" in norm:
                found_type = orig
                break
            elif kind == "noise" and "noise" in norm:
                found_type = orig
                break
            elif kind == "shuffle" and "shuffle" in norm:
                found_type = orig
                break
        if found_type is not None:
            ordered_corr_info.append((kind, found_type))

    if not ordered_corr_info:
        print("No recognized corruption_types (dropout/noise/shuffle) found in CSV.")
        return

    n_panels = len(ordered_corr_info)

    # Figure + layout
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(13, 3.6) if n_panels == 3 else (9, 3.6),
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

    handles_for_legend = []
    labels_for_legend = []

    # Plot each corruption type into its panel
    for ax, (kind, corr_type) in zip(axes, ordered_corr_info):
        df_corr = df[df["corruption_type"] == corr_type].copy()
        if df_corr.empty:
            continue

        # Panel background
        ax.set_facecolor("#F4F6FA")
        ax.set_axisbelow(True)

        for mv, df_mv in df_corr.groupby("model_variant"):
            col = color_map[mv]

            # Group by corruption_param and compute means across folds
            grouped = df_mv.groupby("corruption_param")
            xs_means = []
            ys_means = []
            ys_stds = []

            for cp, df_cp in grouped:
                xs_means.append(float(cp))
                ys_means.append(float(df_cp["plot_value"].mean()))
                ys_stds.append(float(df_cp["plot_value"].std(ddof=0)))

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

            display_name = get_display_name(mv)

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
            # Error bars if we have std
            if np.any(ys_stds > 0):
                ax.errorbar(
                    xs_means,
                    ys_means,
                    yerr=ys_stds,
                    fmt="none",
                    marker=None,
                    ecolor=col,
                    alpha=0.4,
                    linewidth=0.8,
                    capsize=4,
                    capthick=0.8,
                    zorder=1,
                )

            if display_name not in labels_for_legend:
                labels_for_legend.append(display_name)
                handles_for_legend.append(line)

        # Titles and x-axis labels per panel
        if kind == "dropout":
            ax.set_title("Modality Dropout", pad=6, color="#1A1A1A")
            ax.set_xlabel("Dropout Probability", fontsize=10)
            desired_ticks = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
        elif kind == "noise":
            ax.set_title("Modality Noise", pad=6, color="#1A1A1A")
            ax.set_xlabel("Relative Gaussian Noise Scale (σ_noise / σ_feature)", fontsize=10)
            desired_ticks = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
        else:  # "shuffle"
            ax.set_title("Modality Shuffle", pad=6, color="#1A1A1A")
            ax.set_xlabel("Shuffle Probability", fontsize=10)
            desired_ticks = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]

        cp_vals = np.unique(df_corr["corruption_param"].values.astype(float))
        valid_ticks = [t for t in desired_ticks if np.any(np.isclose(cp_vals, t, atol=1e-8))]

        ax.set_xticks(valid_ticks)
        ax.set_xticklabels([f"{t:.2f}" for t in valid_ticks])

        ax.yaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.7, color="#BFC7D5")
        ax.xaxis.grid(False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#A9B2C3")

        # Y-axis ticks at every 0.01 (as requested)
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Shared Y label
    metric_display = format_metric_name(metric_name)
    ylabel = f"{metric_display} (macro)\n(averaged over constructs)"
    fig.text(
        0.015,
        0.5,
        ylabel,
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=10,
    )

    # Shared legend
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
            prop={'weight': 'bold', 'size': 11},
        )

    plt.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.22, wspace=0.20)

    out_path_png = os.path.join(
        output_dir, f"corruption_trends__{metric_name}__reliability_switch.png"
    )
    out_path_pdf = os.path.join(
        output_dir, f"corruption_trends__{metric_name}__reliability_switch.pdf"
    )
    fig.savefig(out_path_png, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    print(f"Saved: {out_path_png}")
    print(f"Saved: {out_path_pdf}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_noise_two_splits(
    df_raw: pd.DataFrame,
    output_dir: str,
    metric_name: str = "f1_macro",
    modality_filter: str = "random_single",
    show: bool = False,
) -> None:
    """
    Specialized visualization: Modality Noise only, with two panels:
      - Left: individual split
      - Right: group split
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build per-fold aggregated F1 for each split separately
    df_ind = build_per_fold_f1_df(
        df_raw,
        split_filter="individual",
        modality_filter=modality_filter,
        metric_name=metric_name,
    )
    df_grp = build_per_fold_f1_df(
        df_raw,
        split_filter="group",
        modality_filter=modality_filter,
        metric_name=metric_name,
    )

    # Filter down to noise corruption only
    def _filter_noise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["corr_norm"] = (
            df["corruption_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("-", " ")
            .str.replace("_", " ")
        )
        noise_mask = df["corr_norm"].str.contains("noise")
        return df[noise_mask].copy()

    df_ind = _filter_noise(df_ind)
    df_grp = _filter_noise(df_grp)

    if df_ind.empty and df_grp.empty:
        print("No 'noise' corruption_type rows found for either split.")
        return

    # Figure layout: two panels side-by-side, independent y-limits.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    fig.patch.set_facecolor("#FAFAFB")

    # Build shared color map across both splits
    all_variants = sorted(
        set(df_ind["model_variant"].unique().tolist())
        | set(df_grp["model_variant"].unique().tolist())
    )
    color_map: Dict[str, str] = {}
    palette_index = 0
    for mv in all_variants:
        key = mv.lower()
        if key in COLOR_OVERRIDES:
            color_map[mv] = COLOR_OVERRIDES[key]
        else:
            color_map[mv] = PALETTE[palette_index % len(PALETTE)]
            palette_index += 1

    def _plot_one_split(ax, df_split: pd.DataFrame, split_name: str):
        if df_split.empty:
            ax.set_visible(False)
            return [], []

        ax.set_facecolor("#F4F6FA")
        ax.set_axisbelow(True)

        handles = []
        labels = []

        for mv, df_mv in df_split.groupby("model_variant"):
            col = color_map[mv]

            # Group by corruption_param to get mean/std over folds
            grouped = df_mv.groupby("corruption_param")
            xs = []
            ys = []
            ys_std = []
            for cp, df_cp in grouped:
                xs.append(float(cp))
                ys.append(float(df_cp["value"].mean()))
                ys_std.append(float(df_cp["value"].std(ddof=0)))
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            ys_std = np.asarray(ys_std, dtype=float)
            if len(xs) == 0:
                continue
            order = np.argsort(xs)
            xs = xs[order]
            ys = ys[order]
            ys_std = ys_std[order]

            display_name = get_display_name(mv)
            line = ax.plot(
                xs,
                ys,
                color=col,
                linestyle="-",
                linewidth=2.0,
                label=display_name,
                zorder=3,
            )[0]
            ax.scatter(
                xs,
                ys,
                color=col,
                alpha=0.98,
                s=55,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            if np.any(ys_std > 0):
                ax.errorbar(
                    xs,
                    ys,
                    yerr=ys_std,
                    fmt="none",
                    marker=None,
                    ecolor=col,
                    alpha=0.4,
                    linewidth=0.8,
                    capsize=4,
                    capthick=0.8,
                    zorder=1,
                )

            if display_name not in labels:
                labels.append(display_name)
                handles.append(line)

        # Axis styling
        ax.set_title(
            f"Modality Noise ({split_name.capitalize()})", pad=6, color="#1A1A1A"
        )
        ax.set_xlabel("Relative Gaussian Noise Scale (σ_noise / σ_feature)", fontsize=10)

        cp_vals = (
            df_split["corruption_param"].astype(float).unique()
            if not df_split.empty
            else np.array([])
        )
        desired_ticks = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
        valid_ticks = [
            t for t in desired_ticks if np.any(np.isclose(cp_vals, t, atol=1e-8))
        ]
        ax.set_xticks(valid_ticks)
        ax.set_xticklabels([f"{t:.2f}" for t in valid_ticks])

        # Y-axis: automatic per-panel limits + ticks
        y_vals = df_split["value"].to_numpy(dtype=float)
        y_vals = y_vals[np.isfinite(y_vals)]
        if y_vals.size > 0:
            y_min = float(y_vals.min())
            y_max = float(y_vals.max())
            data_range = y_max - y_min
            margin = max(0.01, data_range * 0.05) if data_range > 0 else 0.02
            lo = max(0.0, y_min - margin)
            hi = min(1.0, y_max + margin)
            if hi <= lo:
                hi = lo + 0.10
            ax.set_ylim(lo, hi)

        # Y-axis ticks at every 0.01 (as requested)
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax.yaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.7, color="#BFC7D5")
        ax.xaxis.grid(False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#A9B2C3")

        return handles, labels

    handles_all, labels_all = [], []
    for ax, (split_df, split_name) in zip(
        axes, [(df_ind, "individual"), (df_grp, "group")]
    ):
        h, l = _plot_one_split(ax, split_df, split_name)
        handles_all.extend(h)
        labels_all.extend(l)

    # Shared Y label
    metric_display = format_metric_name(metric_name)
    ylabel = f"{metric_display} (macro)\n(averaged over constructs)"
    fig.text(
        0.001,
        0.5,
        ylabel,
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=10,
    )

    # Shared legend
    if handles_all and labels_all:
        # Deduplicate by label
        uniq_handles = []
        uniq_labels = []
        for h, lab in zip(handles_all, labels_all):
            if lab not in uniq_labels:
                uniq_labels.append(lab)
                uniq_handles.append(h)
        fig.legend(
            uniq_handles,
            uniq_labels,
            loc="upper center",
            ncol=min(len(uniq_labels), 5),
            frameon=False,
            bbox_to_anchor=(0.5, 0.97),
            columnspacing=2.5,
            handlelength=1.8,
            prop={'weight': 'bold', 'size': 11},
        )

    plt.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.22, wspace=0.20)

    out_path_png = os.path.join(
        output_dir, f"corruption_trends__{metric_name}__noise_two_splits.png"
    )
    out_path_pdf = os.path.join(
        output_dir, f"corruption_trends__{metric_name}__noise_two_splits.pdf"
    )
    fig.savefig(out_path_png, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    print(f"Saved: {out_path_png}")
    print(f"Saved: {out_path_pdf}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot reliability-switch corruption sweeps with per-fold points and "
            "straight-line connections per fusion variant.\n"
            "Produces a figure with 3 horizontal subplots "
            "(Modality Dropout, Modality Noise, Modality Shuffle)."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing reliability results (e.g. resultsNEW or resultsNEW/kfold).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots_reliability_switch",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Comma-separated list of variant names to include "
            "(e.g. 'glrx,uniform_avg,concat_mlp'). "
            "If omitted, all variants present under root-dir are used."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1_macro",
        help="Metric name to aggregate and plot (default: f1_macro).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["individual", "group", "all"],
        default="group",
        help="Split to visualize: 'individual', 'group', or 'all' (averaged across both).",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="random_single",
        help="Modality filter for stress_tests.csv (default: random_single).",
    )
    parser.add_argument(
        "--noise-only",
        action="store_true",
        help=(
            "If set, plot only Modality Noise as two panels "
            "(individual split and group split) instead of all corruption types."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, display plots interactively.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.variants is not None:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = None

    raw = collect_stress_results(args.root_dir, variants=variants)

    if args.noise_only:
        plot_noise_two_splits(
            raw,
            output_dir=args.outdir,
            metric_name=args.metric,
            modality_filter=args.modality,
            show=args.show,
        )
    else:
        per_fold_df = build_per_fold_f1_df(
            raw,
            split_filter=args.split,
            modality_filter=args.modality,
            metric_name=args.metric,
        )

        plot_corruption_trends(
            per_fold_df,
            output_dir=args.outdir,
            metric_name=args.metric,
            show=args.show,
        )


if __name__ == "__main__":
    main()


