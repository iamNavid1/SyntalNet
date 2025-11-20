# SOSEX Experiments Evaluation

Evaluation suite for different SOSEX model variants. This script runs `evaluate.py` on multiple model variants and collects metrics for all heads.

## Usage

```bash
python experiments/sosex_experiments/runner.py \
    --configs configs/EXPT_SoSEX_w-mp-fus_wo-grp-cls.yaml \
              configs/EXPT_SoSEX_wo-mp-fus_w-grp-cls.yaml \
              configs/EXPT_SoSEX_wo-mp-fus_wo-grp-cls.yaml \
    --output-dir results/sosex_experiments/
```

### Arguments

- `--configs`: List of config YAML files for each variant (required)
- `--checkpoint-base-dir`: Base directory for checkpoints (optional, will use checkpoint_dir from config if not provided)
- `--output-dir`: Directory to save results (default: `results/sosex_experiments`)
- `--device`: Device to use (default: auto-detect)

## Checkpoint Directory Structure

The script automatically extracts the checkpoint directory from each config file's `logging.checkpoint_dir` field. The checkpoint directory structure should match what `evaluate.py` expects:

**For kfold/logo splits:**
```
checkpoint_dir/
├── kfold/fold_00/best.pth (or latest.pth, epoch_100.pth, checkpoint.pth)
├── kfold/fold_01/best.pth
└── ...
```

**For item/group splits:**
```
checkpoint_dir/
├── best.pth (or latest.pth, epoch_100.pth, checkpoint.pth)
└── ...
```

## Output

- `sosex_experiments_results_per_fold.csv`: Per-fold metrics for all heads
- `sosex_experiments_results_aggregated.csv`: Aggregated metrics (mean/std/min/max/median/q1/q3) for all heads
- `sosex_experiments.log`: Execution log

## Metrics Collected

For each head, the following metrics are collected:
- **AUPRC macro**: Area Under Precision-Recall Curve (macro average)
- **AUROC macro**: Area Under ROC Curve (macro average)
- **balanced_accuracy**: Balanced accuracy (macro average)
- **F1 macro**: F1 score (macro average)


