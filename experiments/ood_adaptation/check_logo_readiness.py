from __future__ import annotations

import os
import sys
import glob
import re
import argparse
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def discover_group_ids(root_dir: str, modalities: list[str]) -> list[int]:
    """Discover group IDs from dataset."""
    example_mod = modalities[0]
    pattern_csv = os.path.join(root_dir, example_mod, "Group_*.csv")
    pattern_json = os.path.join(root_dir, example_mod, "Group_*.json")
    
    files = glob.glob(pattern_csv) + glob.glob(pattern_json)
    group_ids = set()
    
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"Group_(\d+)\.(csv|json)$", base)
        if m:
            group_ids.add(int(m.group(1)))
    
    return sorted(group_ids)


def check_logo_checkpoint(checkpoint_base_dir: str, fold_idx: int) -> tuple[bool, str | None]:
    """Check if checkpoint exists for a specific LOGO fold."""
    fold_dir = os.path.join(checkpoint_base_dir, "logo", f"fold_{fold_idx:02d}")
    
    if not os.path.isdir(fold_dir):
        return False, f"Directory not found: {fold_dir}"
    
    # Check for any checkpoint
    checkpoint_found = False
    checkpoint_path = None
    
    for name in ["best.pth", "latest.pth", "epoch_100.pth"]:
        path = os.path.join(fold_dir, name)
        if os.path.isfile(path):
            checkpoint_found = True
            checkpoint_path = path
            break
    
    if not checkpoint_found:
        # Try epoch_*.pth files
        epoch_files = glob.glob(os.path.join(fold_dir, "epoch_*.pth"))
        if epoch_files:
            def extract_epoch(p):
                m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(p))
                return int(m.group(1)) if m else -1
            epoch_files.sort(key=extract_epoch)
            checkpoint_path = epoch_files[-1]
            checkpoint_found = True
    
    if checkpoint_found:
        return True, checkpoint_path
    else:
        return False, f"No checkpoint found in {fold_dir}"


def main():
    parser = argparse.ArgumentParser(
        description="Check if LOGO training is complete and ready for OOD adaptation experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML used for LOGO training"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Base directory containing LOGO checkpoints"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Check split mode
    split_cfg = cfg.get("dataset", {}).get("split", {})
    split_mode = split_cfg.get("mode", "item")
    
    if split_mode != "logo":
        print(f"ERROR: Config split mode is '{split_mode}', expected 'logo'")
        print(f"Please use a config with split.mode: 'logo'")
        return 1
    
    # Discover groups
    root_dir = cfg["dataset"]["args"]["root_dir"]
    modalities = cfg["dataset"]["args"]["modalities"]
    all_groups = discover_group_ids(root_dir, modalities)
    
    print("=" * 80)
    print("LOGO Training Readiness Check")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Number of groups: {len(all_groups)}")
    print(f"Groups: {all_groups}")
    print("")
    
    # Check each fold
    all_ready = True
    results = []
    
    for group_id in all_groups:
        ready, info = check_logo_checkpoint(args.checkpoint_dir, group_id)
        results.append((group_id, ready, info))
        
        if ready:
            status = "✓ READY"
        else:
            status = "✗ MISSING"
            all_ready = False
        
        print(f"Fold {group_id:02d}: {status}")
        if ready:
            print(f"  Checkpoint: {info}")
        else:
            print(f"  Issue: {info}")
        print("")
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    ready_count = sum(1 for _, ready, _ in results if ready)
    print(f"Ready: {ready_count}/{len(all_groups)} folds")
    
    if all_ready:
        print("")
        print("✓ All LOGO folds are ready for OOD adaptation experiments!")
        print("")
        print("You can now run:")
        print(f"  python experiments/ood_adaptation/runner.py \\")
        print(f"      --config {args.config} \\")
        print(f"      --checkpoint-dir {args.checkpoint_dir} \\")
        print(f"      --output-dir results/ood_adaptation")
        return 0
    else:
        print("")
        print("✗ Some LOGO folds are missing checkpoints.")
        print("Please complete LOGO training before running OOD adaptation experiments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

