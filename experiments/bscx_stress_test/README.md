# Stress Test Evaluation Suite

This directory contains a comprehensive evaluation suite for stress-testing multi-channel fusion modules in SyntalNet.

## Overview

The stress test suite evaluates model variants under various corruption scenarios to demonstrate the robustness advantages of BSC_X over baseline fusion methods.

## Structure

- `corruptions.py`: Corruption functions (stream_dropout, channel_dropout, temporal_band_mask, misalignment_jitter, energy_imbalance)
- `model_wrapper.py`: Wrapper for injecting corruptions into model forward pass
- `model_loader.py`: Utilities for loading models with different fusion types
- `dataset_builder.py`: 5-fold cross-validation dataset builder
- `evaluator.py`: Evaluation engine with corruption injection
- `metrics_collector.py`: Metrics collection and CSV export
- `runner.py`: Main orchestration script

## Usage

### Basic Usage

```bash
python experiments/stress_test/runner.py \
    --config configs/EXPT_main.yaml \
    --checkpoint-dir ./checkpoints \
    --output-dir ./stress_test_results \
    --n-folds 5
```

### Arguments

- `--config`: Path to YAML config file (used for dataset and model config)
- `--checkpoint-dir`: Directory containing checkpoints for each variant
  - Expected structure: `checkpoint_dir/{variant_name}/best.pth` or `latest.pth`
  - Variants: `bscx`, `proj_only`, `concat_proj`, `uniform_avg`
- `--output-dir`: Directory to save results (CSV files and logs)
- `--n-folds`: Number of folds for cross-validation (default: 5)
- `--variants`: List of variants to evaluate (default: all)
- `--device`: Device to use (cuda/cpu, default: auto-detect)

### Checkpoint Directory Structure

```
checkpoint_dir/
├── bscx/
│   └── best.pth (or latest.pth)
├── proj_only/
│   └── best.pth
├── concat_proj/
│   └── best.pth
└── uniform_avg/
    └── best.pth
```

## Corruption Scenarios

The suite evaluates five corruption scenarios with parameter sweeps:

1. **Stream Dropout**: Drop entire streams with probability `p`
   - Sweep: `[0.0, 0.2, 0.4, 0.6]`

2. **Channel Dropout**: Drop fraction of channels per stream
   - Sweep: `[0.0, 0.25, 0.5, 0.75]`

3. **Temporal Band Mask**: Zero contiguous temporal band
   - Sweep: `[0.0, 0.2, 0.4, 0.6]` (fraction of time dimension)

4. **Misalignment Jitter**: Shift streams along time dimension
   - Sweep: `[0, 5, 10, 20]` (max shift in time steps)

5. **Energy Imbalance**: Scale one stream by factor
   - Sweep: `[1.0, 0.5, 2.0, 3.0]`

## Output

The evaluation produces:

1. **CSV Files**:
   - `stress_test_results_per_fold.csv`: Per-fold metrics for all runs
   - `stress_test_results_aggregated.csv`: Mean/std across folds

2. **Log File**:
   - `stress_test.log`: Detailed execution log

### CSV Format

**Per-fold CSV** columns:
- `model_variant`: Model variant name
- `corruption_type`: Type of corruption
- `corruption_param`: Parameter value
- `fold`: Fold index (0-4)
- `split`: individual/group
- `head`: Classification head name
- `metric`: Metric name
- `value`: Metric value

**Aggregated CSV** columns:
- Same as above, plus:
- `mean`: Mean across folds
- `std`: Standard deviation
- `min`: Minimum value
- `max`: Maximum value
- `n_folds`: Number of folds

## Metrics Collected

For each classification head, the following metrics are collected:
- `auprc_macro`: Area Under Precision-Recall Curve (macro average)
- `auroc_macro`: Area Under ROC Curve (macro average)
- `balanced_accuracy`: Balanced accuracy
- `f1_macro`: Macro F1 score
- Additional metrics (accuracy, precision, recall, etc.)

## Optimization

The suite optimizes computation by:
1. **Reusing dataloaders**: Each fold's dataloader is built once and reused for all corruption scenarios
2. **Baseline caching**: Baseline (no corruption) results are computed once per fold and reused across all corruption types
3. **Efficient batching**: Uses existing DataLoader infrastructure

## Example Workflow

1. Train models for each variant (bscx, proj_only, concat_proj, uniform_avg)
2. Save checkpoints in the expected directory structure
3. Run stress test evaluation:
   ```bash
   python experiments/stress_test/runner.py \
       --config configs/EXPT_main.yaml \
       --checkpoint-dir ./checkpoints \
       --output-dir ./stress_test_results
   ```
4. Analyze results in CSV files
5. Compare robustness across variants

## Customization

To modify corruption sweeps, edit `CORRUPTION_SWEEPS` in `runner.py`:

```python
CORRUPTION_SWEEPS = {
    "stream_dropout": [0.0, 0.2, 0.4, 0.6],
    # ... modify as needed
}
```

To add new corruption types:
1. Implement function in `corruptions.py`
2. Add to `create_corruption_fn()` in `model_wrapper.py`
3. Add sweep configuration in `runner.py`

