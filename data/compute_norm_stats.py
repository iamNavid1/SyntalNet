import argparse
import yaml
import torch
from typing import Any, Dict, List
from tqdm.auto import tqdm

from dataset import GroupDynamicsDataset
from transforms import STANDARDIZATION_GROUPS, EMBEDDING_DIMS


def compute_stats(dataset: GroupDynamicsDataset, show_progress: bool = True) -> Dict[str, Any]:
    sums: Dict[str, Any] = {}
    sqs: Dict[str, Any] = {}
    counts: Dict[str, Any] = {}

    std_mods = set(STANDARDIZATION_GROUPS.keys()) & set(dataset.modalities)
    embed_mods = set(EMBEDDING_DIMS.keys()) & set(dataset.modalities)

    for mod in std_mods:
        groups = STANDARDIZATION_GROUPS[mod]
        sums[mod] = {g: 0.0 for g in groups}
        sqs[mod] = {g: 0.0 for g in groups}
        counts[mod] = {g: 0 for g in groups}

    for mod in embed_mods:
        dim = EMBEDDING_DIMS[mod]
        sums[mod] = torch.zeros(dim, dtype=torch.float64)
        sqs[mod] = torch.zeros(dim, dtype=torch.float64)
        counts[mod] = 0

    it = range(len(dataset))
    if show_progress:
        it = tqdm(it, total=len(dataset), desc="Computing stats", unit="sample")

    for i in it:
        data, _, _, _ = dataset[i]
        for mod in std_mods:
            groups = STANDARDIZATION_GROUPS[mod]
            if mod not in data:
                continue
            tensors = data[mod]
            if not isinstance(tensors, list):
                tensors = [tensors]
            for tensor in tensors:
                tensor = tensor.float()
                for grp, idxs in groups.items():
                    vals = tensor[..., idxs].reshape(-1)
                    sums[mod][grp] += vals.sum().item()
                    sqs[mod][grp] += (vals ** 2).sum().item()
                    counts[mod][grp] += vals.numel()
        for mod in embed_mods:
            if mod not in data:
                continue
            tensors = data[mod]
            if not isinstance(tensors, list):
                tensors = [tensors]
            for tensor in tensors:
                tensor = tensor.float()
                vals = tensor.reshape(-1, EMBEDDING_DIMS[mod])
                sums[mod] += vals.sum(dim=0)
                sqs[mod] += (vals ** 2).sum(dim=0)
                counts[mod] += vals.shape[0]

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for mod in std_mods:
        stats[mod] = {}
        for grp in sums[mod]:
            denom = max(counts[mod][grp], 1)
            mean = sums[mod][grp] / denom
            var = max(sqs[mod][grp] / denom - mean ** 2, 0.0)
            stats[mod][grp] = {
                "mean": float(mean), 
                "std": float(var ** 0.5)
            }

    for mod in embed_mods:
        denom = max(counts[mod], 1)
        mean = sums[mod] / denom
        var = torch.clamp(sqs[mod] / denom - mean ** 2, min=0.0)
        stats[mod] = {
            "mean": mean.tolist(),
            "std": var.sqrt().tolist(),
        }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute normalization statistics")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--output", required=True, help="Path to write YAML stats")
    parser.add_argument(
        "--modalities", nargs="+",
        default=['face', 'pose', 'video', 'turns', 'utterance', 'sentiment', 'prosody', 'audio'],
        help="Modalities to include")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm progress bar")
    args = parser.parse_args()

    print(f"[info] modalities: {args.modalities}")
    ds = GroupDynamicsDataset(root_dir=args.root, modalities=args.modalities)
    print(f"[info] num samples: {len(ds)}")

    stats = compute_stats(ds, show_progress=not args.no_progress)

    with open(args.output, "w") as f:
        yaml.safe_dump(stats, f)
    print(f"[ok] wrote stats to {args.output}")


if __name__ == "__main__":
    main()
