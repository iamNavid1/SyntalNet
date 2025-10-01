from __future__ import annotations

import os
import sys
import yaml
from typing import Dict, List
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import GroupDynamicsDataset


def compute_class_counts(
    dataset: Dataset,
    individual_heads: List[str],
    group_heads: List[str],
) -> Dict[str, Dict[str, List[int]]]:

    counts: Dict[str, Dict[str, List[int]]] = {
        "individual": {h: [0, 0, 0] for h in individual_heads},
        "group": {h: [0, 0, 0] for h in group_heads},
    }

    for i in range(len(dataset)):
        _, _, labels, _ = dataset[i]
        if "individual" in labels:
            indiv = labels["individual"]  # shape: [persons, heads]
            for j, head in enumerate(individual_heads):
                lbls = indiv[:, j].view(-1)
                for c in range(3):
                    counts["individual"][head][c] += int((lbls == c).sum())
        if "group" in labels:
            grp = labels["group"]  # shape: [heads]
            for j, head in enumerate(group_heads):
                lbl = int(grp[j])
                if lbl >= 0:
                    counts["group"][head][lbl] += 1
    return counts


def make_yaml():
    project_root = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(project_root, "configs", "class_counts.yaml")
    dataset_root = os.path.join(project_root, "dataset")
    all_counts = {}

    for lbl_type in ["label", "kernel", "ema", "kalman", "threshold"]:
        ds = GroupDynamicsDataset(
            root_dir=dataset_root,
            modalities=["face"],
            label_type=lbl_type,
        )
        counts = compute_class_counts(
            ds,
            ["Engagement", "Lead"],
            ["Synchrony", "Confidence", "Transition"],
        )

        # print(lbl_type)
        # for head in counts['individual']:
        #     print(head, counts['individual'][head])
        # for head in counts['group']:
        #     print(head, counts['group'][head])
        # print("\n--------------------------------\n")
        all_counts[lbl_type] = counts

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(all_counts, f, default_flow_style=False, sort_keys=False)
        

if __name__ == "__main__":
    make_yaml()