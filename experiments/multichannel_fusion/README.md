# BSCX Stress Test Evaluation

Stress-testing evaluation for BSC_X multi-channel fusion and baseline variants under corruption scenarios.

## Usage

```bash
python experiments/multichannel_fusion/runner.py \
    --config configs/EXPT_main.yaml \
    --checkpoint-dir ./checkpoints \
    --output-dir ./stress_test_results
```

### Arguments

- `--config`: Path to YAML config file
- `--checkpoint-dir`: Directory containing model checkpoints
- `--output-dir`: Directory to save results
- `--variants`: List of variants to evaluate (default: all)
- `--device`: Device to use (default: auto-detect)

## Checkpoint Directory Structure

**For kfold/logo splits:**
```
checkpoint_dir/
├── bscx/kfold/fold_00/best.pth (or latest.pth, epoch_100.pth, checkpoint.pth)
├── bscx/kfold/fold_01/best.pth
├── proj_only/kfold/fold_00/best.pth
└── ...
```

**For item/group splits:**
```
checkpoint_dir/
├── bscx/best.pth (or latest.pth, epoch_100.pth, checkpoint.pth)
├── proj_only/best.pth
├── concat_proj/best.pth
└── uniform_avg/best.pth
```

## Output

- `stress_test_results_per_fold.csv`: Per-fold metrics
- `stress_test_results_aggregated.csv`: Aggregated metrics (mean/std)
- `stress_test.log`: Execution log

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