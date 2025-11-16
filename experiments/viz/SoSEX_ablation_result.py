# Fixed-layout reproduction of the reference figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------- DATA ----------------
labels = ["Group\nConfidence","Group\nSynchrony","Group\nTransition","Individual\nEngagement","Individual\nLead"]

legend_labels = [
    r"$\mathbf{-}$SoSE-X, $\mathbf{-}$Group Head",
    r"$\mathbf{-}$SoSE-X, $\mathbf{+}$Group Head",
    r"$\mathbf{+}$SoSE-X, $\mathbf{-}$Group Head",
    r"$\mathbf{+}$SoSE-X, $\mathbf{+}$Group Head"
]

metrics_data = {
    "AUPRC": np.array([[0.498,0.550,0.644,0.691],[0.493,0.532,0.642,0.666],[0.497,0.591,0.630,0.658],[0.508,0.532,0.665,0.653],[0.468,0.482,0.613,0.597]]),
    "AUROC": np.array([[0.704,0.743,0.823,0.843],[0.707,0.742,0.800,0.819],[0.697,0.768,0.808,0.827],[0.741,0.761,0.864,0.857],[0.692,0.701,0.813,0.804]]),
    "Accuracy": np.array([[0.509,0.538,0.633,0.669],[0.499,0.498,0.636,0.661],[0.498,0.539,0.632,0.635],[0.534,0.550,0.688,0.676],[0.499,0.490,0.618,0.622]]),
    "F1 Score": np.array([[0.514,0.545,0.635,0.668],[0.504,0.509,0.634,0.659],[0.500,0.546,0.626,0.630],[0.533,0.544,0.668,0.653],[0.494,0.488,0.607,0.606]]),
}

# ---------------- STYLE ----------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 350,      # lock on-screen dpi
    "savefig.dpi": 350,     # lock exported dpi
    "axes.linewidth": 1.2,
})

palette = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16"]

# ---------------- GEOMETRY (exact) ----------------
num_labels = len(labels)
num_conditions = len(legend_labels)
bar_width = 0.20          # exact width from reference
inner_gap = 0.05          # gap inside a block
group_gap = 1.0           # gap between blocks

# x-positions
x_positions = []
for i in range(num_labels):
    base = i * (num_conditions*bar_width + (num_conditions-1)*inner_gap + group_gap)
    for j in range(num_conditions):
        x_positions.append(base + j*(bar_width + inner_gap))
x_positions = np.array(x_positions).reshape(num_labels, num_conditions)
group_centers = x_positions.mean(axis=1)
block_width = num_conditions*bar_width + (num_conditions-1)*inner_gap

def compute_ylim(values, margin_low=0.03, margin_high=0.03):
    vmin, vmax = float(values.min()), float(values.max())
    lo = max(0.0, vmin - margin_low); hi = min(1.0, vmax + margin_high)
    lo = np.floor(lo*20)/20.0; hi = np.ceil(hi*20)/20.0
    if hi - lo < 0.20: hi = min(1.0, lo + 0.20)
    return lo, hi

def place_value_labels(ax, bars, min_sep=0.0):
    # Keep value labels readable and on top
    centers = np.array([b.get_x() + b.get_width()/2 for b in bars])
    heights = np.array([b.get_height() for b in bars])
    order = np.argsort(centers)
    prev_y, sign = None, 1
    for idx in order:
        x, y = centers[idx], heights[idx]
        label_y = y + 0.006
        if prev_y is not None and abs(label_y - prev_y) < min_sep:
            label_y = prev_y + sign*(min_sep - abs(label_y - prev_y))
            sign *= -1
        ax.text(x, label_y, f"{y:.3f}", ha="center", va="bottom",
                fontsize=8, zorder=10, clip_on=False)
        prev_y = label_y

# ---------------- FIGURE ----------------
fig, axes = plt.subplots(1, 4, figsize=(21, 3.6), sharey=False)
fig.patch.set_facecolor('#FAFAFB')
custom_ylim = {"AUPRC": (0.40, 0.75)}

for ax, (metric_name, metric_matrix) in zip(axes, metrics_data.items()):
    y_lo, y_hi = custom_ylim.get(metric_name, compute_ylim(metric_matrix))

    # exact-aligned background band for each block
    for i in range(num_labels):
        ax.add_patch(Rectangle((x_positions[i,0]-0., y_lo), block_width+0., y_hi - y_lo,
                               facecolor='#F4F6FA', edgecolor='none', zorder=0, alpha=1.0))

    # bars
    all_bars = []
    for c in range(num_conditions):
        bars = ax.bar(x_positions[:, c], metric_matrix[:, c],
                      width=bar_width, color=palette[c], edgecolor='white',
                      linewidth=1.2, alpha=0.97, zorder=3,
                      label=legend_labels[c])
        all_bars.append(bars)

    # labels with collision-avoidance per block
    for i in range(num_labels):
        place_value_labels(ax, [all_bars[c][i] for c in range(num_conditions)], min_sep=0.0)

    ax.set_title(metric_name, pad=6, fontsize=13, weight='bold', color='#1A1A1A')
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(group_centers); ax.set_xticklabels(labels, ha="center", fontsize=10, color='#2C2C2C')
    ax.yaxis.grid(True, linestyle=':', linewidth=0.8, alpha=0.7, color='#BFC7D5', zorder=1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#A9B2C3'); ax.spines['bottom'].set_color('#A9B2C3')
    # ax.set_ylabel(metric_name, labelpad=4)

# one-line shared legend + title
handles, _ = axes[0].get_legend_handles_labels()
fig.legend(handles[:4], legend_labels, loc="upper center", ncol=4,
           frameon=False, bbox_to_anchor=(0.5, 1.03),
           columnspacing=2.8, handlelength=1.8)
# fig.suptitle("Performance by Label qand Condition", y=1.10, fontsize=15, weight='bold')

plt.subplots_adjust(left=0.04, right=0.995, top=0.86, bottom=0.22, wspace=0.2)

# SAVE with locked DPI so proportions don’t change
plt.savefig("metrics_fixed_layout.png", bbox_inches="tight")
plt.savefig("metrics_fixed_layout.pdf", bbox_inches="tight")
plt.show()
