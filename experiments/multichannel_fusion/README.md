# BSCX Stress Test Evaluation

Stress-testing evaluation for BSC_X multi-channel fusion and baseline variants under corruption scenarios.

## Usage

```bash
python experiments/multichannel_fusion/runner.py \
    --configs configs/EXPT_BSCX_bscx.yaml \
              configs/EXPT_BSCX_concat-proj.yaml \
              configs/EXPT_BSCX_uniform-avg.yaml \
    --output-dir ./stress_test_results
```

### Arguments

- `--configs`: Paths to YAML config files (one per variant).
- `--output-dir`: Directory to save results
- `--variants`: List of variant names to evaluate (default: all configs). If provided, filters configs.
- `--device`: Device to use (default: auto-detect)

## Checkpoint Directory Structure

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

- `stress_test_results_per_fold.csv`: Per-fold raw metrics
- `stress_test_results_per_construct.csv`: Per-construct aggregated stats (over folds)
- `stress_test_results_all_constructs.csv`: All-constructs aggregated stats (by split and combined)
- `stress_test.log`: Execution log

**Metrics collected**: accuracy, f1_macro, auroc_macro, auprc_macro

**Statistics computed**: mean, std, min, max, median, q1, q3, n_folds

## Model Variants

- `bscx`: Full BSC_X model
- `proj_only`: Projection only
- `concat_proj`: Concatenation + projection
- `uniform_avg`: Uniform averaging baseline

## Corruption Scenarios

- Stream dropout: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
- Channel dropout: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
- Temporal band mask: [0.0, 0.15, 0.30, 0.45, 0.6, 0.75]
- Misalignment jitter: [0, 5, 10, 15, 20, 25]
- Noise: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]