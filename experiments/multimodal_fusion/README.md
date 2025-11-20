# GLRX Stress Test Evaluation

Stress-testing evaluation for GLR_X multimodal fusion and baseline variants under corruption scenarios.

## Usage

```bash
python experiments/multimodal_fusion/runner.py \
    --configs configs/EXPT_GLRX_glrx.yaml \
              configs/EXPT_GLRX_uniform-avg.yaml \
              configs/EXPT_GLRX_gated_sum.yaml \
              configs/EXPT_GLRX_pairwise.yaml \
              configs/EXPT_GLRX_concat_mlp.yaml \
    --output-dir multimodal_fusion_results/multimodal_fusion/
```

### Arguments

- `--configs`: Paths to YAML config files (one per variant).
- `--output-dir`: Directory to save results
- `--variants`: List of variant names to evaluate (default: all configs). If provided, filters configs.
- `--no-allocation`: Skip allocation tracking experiment
- `--device`: Device to use (default: auto-detect)

## Checkpoint Directory Structure

**For kfold/logo splits:**
```
checkpoint_dir/
├── glrx/kfold/fold_00/best.pth (or latest.pth, epoch_100.pth, checkpoint.pth)
├── glrx/kfold/fold_01/best.pth
├── uniform_avg/kfold/fold_00/best.pth
└── ...
```

**For item/group splits:**
```
checkpoint_dir/
├── glrx/best.pth (or latest.pth, epoch_100.pth, checkpoint.pth)
├── uniform_avg/best.pth
├── gated_sum/best.pth
├── pairwise/best.pth
└── concat_mlp/best.pth
```

## Output

- `multimodal_fusion_results_per_fold.csv`: Per-fold metrics
- `multimodal_fusion_results_aggregated.csv`: Aggregated metrics (mean/std)
- `allocation_tracking/`: Allocation tracking results (if enabled)
- `multimodal_fusion.log`: Execution log

## Model Variants

- `glrx`: Full GLRX multimodal fusion
- `uniform_avg`: Uniform averaging baseline
- `gated_sum`: Gated sum only
- `pairwise`: Pairwise interactions only
- `concat_mlp`: Concatenation + MLP baseline

## Corruption Scenarios

- Modality dropout: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
- Modality noise: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
- Modality shuffle: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
- Modality rescale: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

## Modalitty Allocation vs SNR

- Allocation noise levels: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
