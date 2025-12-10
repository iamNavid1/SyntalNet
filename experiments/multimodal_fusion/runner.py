from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.multimodal_fusion.config import load_experiment_config
from experiments.multimodal_fusion.experiment import ReliabilityExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reliability-switch multimodal fusion experiments"
    )
    parser.add_argument(
        "--config", required=True, help="Path to reliability experiment YAML config"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both"],
        default="both",
        help="Whether to train, evaluate, or do both.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold index to run. Defaults to all folds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., cuda:0 or cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    device = args.device or cfg.device
    runner = ReliabilityExperimentRunner(cfg, device=device)

    folds = [args.fold] if args.fold is not None else list(range(cfg.n_folds))

    for fold_idx in folds:
        if args.mode == "train":
            # train only (no evaluation or stress tests)
            runner.train_fold(fold_idx, run_evaluation=False)
        elif args.mode == "eval":
            runner.evaluate_fold(fold_idx)
        elif args.mode == "both":
            # train + evaluation + stress tests in one go
            runner.train_fold(fold_idx, run_evaluation=True)


if __name__ == "__main__":
    main()

