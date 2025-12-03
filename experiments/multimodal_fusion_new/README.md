# Reliability-Switch Multimodal Fusion Experiments

This directory contains infrastructure for training and evaluating different multimodal fusion variants (GLR-X, UniformAvg, ConcatMLP) under a **reliability-switch** regime, where modalities are selectively corrupted during training to test robustness.

## Overview

The key idea is to:
1. **Freeze** all SyntalNet encoders and branch mixers at their trained checkpoints
2. **Replace** the multimodal fusion module with different variants
3. **Train** only the fusion module and classification heads using a reliability-switch regime:
   - For each sample, exactly one modality is randomly corrupted (dropout/noise/shuffle)
   - Corruption probability: 70% per sample
   - Corruption strength: randomly sampled from [0.1, 0.5]
4. **Evaluate** trained models under systematic stress tests with increasing corruption

## Directory Structure

```
multimodal_fusion_new/
├── __init__.py
├── README.md                      # This file
├── fusion_variants.py             # Clean interface for fusion modules
├── model_loader.py                # Load and freeze SyntalNet checkpoints
├── corruptions.py                 # Corruption utilities (dropout, noise, shuffle)
├── reliability_trainer.py         # Training with reliability-switch augmentation
├── evaluator.py                   # Stress test evaluation
├── config.py                      # Configuration management
├── runner.py                      # Main entry point
├── run_reliability_experiments.sh # Shell script to run all variants
└── configs/
    ├── reliability_glrx.yaml      # GLR-X experiment config
    ├── reliability_uniform_avg.yaml  # UniformAvg baseline config
    └── reliability_concat_mlp.yaml   # ConcatMLP baseline config
```

## Requirements

- Trained SyntalNet checkpoints for all folds (e.g., in `checkpoints/_FINAL_80-20_NO-SOSEX_Stratified/kfold/fold_XX/`)
- SyntalNet configuration file (e.g., `configs/SyntalNet.yaml`)

## Quick Start

### 1. Run All Experiments

To train and evaluate all fusion variants across all folds:

```bash
bash experiments/multimodal_fusion_new/run_reliability_experiments.sh
```

This will:
- Train GLR-X, UniformAvg, and ConcatMLP on all 5 folds
- Run stress tests for each variant
- Save results to `experiments/multimodal_fusion_reliability_results/`

### 2. Run Single Variant

To run a specific fusion variant:

```bash
python experiments/multimodal_fusion_new/runner.py \
    --config experiments/multimodal_fusion_new/configs/reliability_glrx.yaml \
    --mode both \
    --device cuda:0
```

Options:
- `--mode`: `train`, `eval`, or `both`
- `--fold`: Specific fold index (default: run all folds)
- `--device`: Device to use (default: `cuda:0`)

### 3. Train Only

```bash
python experiments/multimodal_fusion_new/runner.py \
    --config experiments/multimodal_fusion_new/configs/reliability_glrx.yaml \
    --mode train \
    --fold 0
```

### 4. Evaluate Only

After training, evaluate on stress tests:

```bash
python experiments/multimodal_fusion_new/runner.py \
    --config experiments/multimodal_fusion_new/configs/reliability_glrx.yaml \
    --mode eval \
    --fold 0
```

## Configuration

Each YAML config contains:

```yaml
# Base SyntalNet checkpoint location
base_checkpoint_dir: "checkpoints/_FINAL_80-20_NO-SOSEX_Stratified"
base_config_path: "configs/SyntalNet.yaml"

# Fusion variant
fusion_variant: "glrx"  # or "uniform_avg", "concat_mlp"
fusion_kwargs:
  rank_pair: 12
  alloc_hidden: 24
  eps_floor: 0.03
  p_drop: 0.1

# Training settings
reliability_training:
  num_epochs: 30
  p_corrupt: 0.7  # Probability of corrupting each sample
  corruption_strengths:
    dropout: [0.1, 0.5]  # Random range for dropout
    noise: [0.1, 0.5]    # Random range for noise
    shuffle: [0.1, 0.5]  # Random range for shuffle
  learning_rate: 5.0e-4
  scheduler: "reduce_on_plateau"
  scheduler_patience: 3

# Stress testing
stress_test:
  corruption_configs:
    dropout: [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
    noise: [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
    shuffle: [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
  per_modality: true  # Also test per-modality corruption

# Cross-validation
cv_mode: "kfold"
n_folds: 5
```

## Output Structure

Results are organized by variant and fold:

```
experiments/multimodal_fusion_reliability_results/
└── kfold/
    ├── glrx/
    │   ├── fold_00/
    │   │   ├── clean_metrics.csv         # Clean performance
    │   │   └── stress_test_results.csv   # Stress test results
    │   ├── fold_01/
    │   │   └── ...
    │   └── aggregated/
    │       ├── stress_test_all_folds.csv    # Combined results
    │       └── stress_test_summary.csv      # Mean/std across folds
    ├── uniform_avg/
    │   └── ...
    └── concat_mlp/
        └── ...
```

### Result CSV Columns

**stress_test_results.csv:**
- `fold`: Fold index
- `variant`: Fusion variant name
- `corruption_type`: Type of corruption (dropout, noise, shuffle)
- `corruption_param`: Corruption strength
- `modality`: Which modality was corrupted (or 'all')
- `split`: Classification split (individual, group)
- `head`: Classification head (Engagement, Valence)
- `metric`: Performance metric (f1_macro, auroc, auprc)
- `value`: Metric value

**clean_metrics.csv:**
- Performance on uncorrupted data
- Per-head metrics for baseline comparison

## Visualization

To visualize stress test results, use the visualization script:

```bash
python experiments/viz/GLRX_ablation_results.py \
    --csv experiments/multimodal_fusion_reliability_results/kfold/glrx/aggregated/stress_test_summary.csv \
    --aggregation-level aggregated \
    --outdir experiments/viz_results/reliability_switch \
    --variants glrx,uniform_avg,concat_mlp
```

This will generate plots showing:
- Performance degradation under increasing corruption
- Comparison across fusion variants
- Per-modality robustness (if enabled)
- Allocation tracking for GLR-X (if available)

## Implementation Details

### Freezing Strategy

- **Frozen**: All encoders, branch mixers (mc_fusion), and branch parameters
- **Trainable**: Multimodal fusion module (mm_fusion) and classification heads
- **Initialization**: Fusion module trained from scratch, classification heads initialized from SyntalNet checkpoint

### Corruption Strategy

During training, for each batch:
1. Forward through frozen branches to get embeddings `z_branch` (list of 3 tensors, each `[B, D]`)
2. For each sample (with probability `p_corrupt = 0.7`):
   - Randomly select ONE modality to corrupt
   - Sample corruption type from mixture: {dropout: 40%, noise: 40%, shuffle: 20%}
   - Apply corruption with specific semantics:
     - **Dropout**: Modality-level scaling (multiply by `1 - dropout_scale`)
     - **Noise**: Per-sample Gaussian noise (std = `noise_level * sample_std`)
     - **Shuffle**: Cross-sample mixing via Bernoulli mask (`shuffle_fraction`)
3. Forward through trainable fusion and classifiers

**Corruption Semantics:**
- **Dropout**: Unlike element-wise dropout, scales the entire embedding by `(1 - dropout_scale)`
- **Noise**: Computed per-sample (not batch-level), preserves sample-specific statistics
- **Shuffle**: Mixes with a different sample from the batch using Bernoulli masking

During evaluation (stress tests):
- No corruption during validation (clean performance)
- Systematic sweeps of corruption strengths for stress testing
- Both uniform (all modalities) and per-modality corruption

### Metrics Tracked

- **Performance**: F1-macro, AUROC, AUPRC per head
- **Allocation** (GLR-X only): Modality weights under corruption
- **Robustness**: Performance degradation slopes under increasing corruption

## Extending

### Adding New Fusion Variants

1. Implement fusion module in `models/multimodal_fusion.py`
2. Register in `fusion_variants.py`:
   ```python
   AVAILABLE_VARIANTS["my_variant"] = MyFusionClass
   ```
3. Create config file: `configs/reliability_my_variant.yaml`
4. Add to shell script

### Custom Corruption Functions

Add new corruption types in `corruptions.py`:
```python
def corrupt_branch_my_corruption(z: torch.Tensor, param: float) -> torch.Tensor:
    # Implement corruption
    return z_corrupted
```

Update `apply_reliability_switch_corruption` to include new type.

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

## Contact

For questions or issues, please contact [your email].

