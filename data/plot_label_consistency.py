import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from collections import defaultdict, Counter
from matplotlib import patches as mpatches

from dataset import GroupDynamicsDataset

COLORS = {2: 'green', 1: 'yellow', 0: 'red', -1: 'lightgray'}

def _majority(vals, tie_break=1):
    nums = [int(v) for v in vals if int(v) in (0,1,2)]
    if not nums:
        return -1
    c = Counter(nums)
    top = max(c.values())
    modes = [k for k,v in c.items() if v == top]
    if len(modes) == 1:
        return modes[0]
    return tie_break if tie_break in modes else sorted(modes)[0]

def _extract_q_labels(labels, tie_break=1):
    g = labels['group'].detach().cpu().numpy() if torch.is_tensor(labels['group']) else np.asarray(labels['group'])
    q1_3 = [int(g[i]) if int(g[i]) in (0,1,2) else -1 for i in range(3)]

    ind = labels['individual']
    ind = ind.detach().cpu().numpy() if torch.is_tensor(ind) else np.asarray(ind)
    q4 = _majority(ind[:,0], tie_break=tie_break)
    q5 = _majority(ind[:,1], tie_break=tie_break)
    return q1_3 + [q4, q5]

def collect_labels_once(dataset, tie_break=1):
    data = defaultdict(lambda: {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': []})
    for i in range(len(dataset)):
        _, _, labels, gid = dataset[i]
        qvals = _extract_q_labels(labels, tie_break=tie_break)
        for k, name in enumerate(['Q1','Q2','Q3','Q4','Q5']):
            data[int(gid)][name].append(qvals[k])

    group_ids = sorted(data.keys())
    max_len = max(max(len(seq) for seq in group.values()) for group in data.values())
    return data, group_ids, max_len

def plot_single_label_from_cache(
    data, group_ids, max_len,
    qname='Q1',
    rect_height=5.0,
    rect_width=1.0,
    row_gap=0.4,
    dpi=150,
    save_path=None
):
    patches, facecolors = [], []
    y_cursor = 0.0
    y_centers = []

    for gid in group_ids:
        seq = data[gid][qname]
        for col, lab in enumerate(seq):
            patches.append(Rectangle((col * rect_width, y_cursor), rect_width, rect_height))
            facecolors.append(COLORS.get(int(lab) if lab in (0,1,2,-1) else -1, 'lightgray'))
        y_centers.append((gid, y_cursor + rect_height/2.0))
        y_cursor += rect_height + row_gap

    fig_w = max(6, 0.12 * max_len * rect_width)
    fig_h = max(4, 0.25 * len(group_ids) + 0.05 * rect_height * len(group_ids))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    coll = PatchCollection(patches, edgecolor='none')
    coll.set_facecolor(facecolors)
    ax.add_collection(coll)

    for gid, y in y_centers:
        ax.text(-0.6 * rect_height, y, f"G{gid:02d}", ha='right', va='center', fontsize=9)

    ax.set_xlim(0, max_len * rect_width)
    ax.set_ylim(-row_gap, y_cursor)
    ax.set_aspect('equal')
    ax.axis('off')

    legend_patches = [
        mpatches.Patch(color=COLORS[2], label='Increasing'),
        mpatches.Patch(color=COLORS[1], label='Staying the same'),
        mpatches.Patch(color=COLORS[0], label='Decreasing'),
        mpatches.Patch(color=COLORS[-1], label='Missing'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', frameon=False, fontsize=9)

    ax.set_title(f"Temporal Consistency — {qname} (rows = Groups)", fontsize=12, pad=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



if __name__ == '__main__':
    out_dir="./label_plots"
    os.makedirs(out_dir, exist_ok=True)

    dataset = GroupDynamicsDataset(
        root_dir="/home/npargoo/Desktop/SyntalNet/dataset/processed",
        modalities=["face"],
        label_tolerance=1,
    )

    data, group_ids, max_len = collect_labels_once(dataset)

    for q in ['Q1','Q2','Q3','Q4','Q5']:
        save_path = os.path.join(out_dir, f"{q}_temporal_consistency.png")
        plot_single_label_from_cache(
            data, group_ids, max_len,
            qname=q,
            rect_height=5,
            rect_width=1,
            row_gap=5,
            dpi=150,
            save_path=save_path
        )
