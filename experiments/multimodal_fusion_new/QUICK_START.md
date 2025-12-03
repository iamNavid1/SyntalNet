# Quick Start Guide

## Prerequisites

1. Trained SyntalNet checkpoints for all folds:
   ```
   checkpoints/_FINAL_80-20_NO-SOSEX_Stratified/kfold/fold_00/
   checkpoints/_FINAL_80-20_NO-SOSEX_Stratified/kfold/fold_01/
   ...
   checkpoints/_FINAL_80-20_NO-SOSEX_Stratified/kfold/fold_04/
   ```

2. SyntalNet config file:
   ```
   configs/SyntalNet.yaml
   ```

## Quick Commands

### 1. Run All Experiments (Recommended)

Train and evaluate all three fusion variants across all 5 folds:

```bash
bash experiments/multimodal_fusion_new/run_reliability_experiments.sh
```

This will take several hours depending on your hardware.

### 2. Run Single Variant

Train and evaluate only GLR-X:

```bash
python experiments/multimodal_fusion_new/runner.py \
    --config experiments/multimodal_fusion_new/configs/reliability_glrx.yaml \
    --mode both \
    --device cuda:0
```

### 3. Run Single Fold

Train and evaluate GLR-X on fold 0 only:

```bash
python experiments/multimodal_fusion_new/runner.py \
    --config experiments/multimodal_fusion_new/configs/reliability_glrx.yaml \
    --mode both \
    --fold 0 \
    --device cuda:0
```

### 4. Train Only (No Evaluation)

```bash
python experiments/multimodal_fusion_new/runner.py \
    --config experiments/multimodal_fusion_new/configs/reliability_glrx.yaml \
    --mode train \
    --device cuda:0
```

### 5. Evaluate Only (After Training)

```bash
python experiments/multimodal_fusion_new/runner.py \
    --config experiments/multimodal_fusion_new/configs/reliability_glrx.yaml \
    --mode eval \
    --device cuda:0
```

## Post-Processing

### Analyze Results

Generate summary tables and statistics:

```bash
python experiments/multimodal_fusion_new/analyze_results.py \
    --results-dir experiments/multimodal_fusion_reliability_results \
    --output-dir experiments/multimodal_fusion_reliability_results/analysis \
    --variants glrx,uniform_avg,concat_mlp
```

This will print tables showing:
- Clean performance comparison
- Robustness metrics (degradation under corruption)
- Sorted by relative degradation (lower = more robust)

### Visualize Results

Generate comparison plots:

```bash
python experiments/multimodal_fusion_new/visualize_results.py \
    --results-dir experiments/multimodal_fusion_reliability_results \
    --output-dir experiments/viz_results/reliability_switch \
    --variants glrx,uniform_avg,concat_mlp \
    --metric f1_macro \
    --plot-allocation
```

This generates:
- `reliability_comparison_f1_macro.png/pdf`: Performance vs corruption strength
- `allocation_under_corruption_glrx.png/pdf`: Allocation tracking for GLR-X

### Filter by Split or Head

To plot only individual-level Engagement:

```bash
python experiments/multimodal_fusion_new/visualize_results.py \
    --results-dir experiments/multimodal_fusion_reliability_results \
    --output-dir experiments/viz_results/reliability_switch \
    --variants glrx,uniform_avg,concat_mlp \
    --metric f1_macro \
    --split individual \
    --head Engagement
```

## Expected Output Structure

```
experiments/multimodal_fusion_reliability_results/
└── kfold/
    ├── glrx/
    │   ├── fold_00/
    │   │   ├── clean_metrics.csv
    │   │   └── stress_test_results.csv
    │   ├── fold_01/
    │   │   └── ...
    │   └── aggregated/
    │       ├── stress_test_all_folds.csv
    │       ├── stress_test_summary.csv
    │       └── clean_metrics_all_folds.csv
    ├── uniform_avg/
    │   └── ...
    └── concat_mlp/
        └── ...
```

## Checkpoint Structure

```
checkpoints/multimodal_fusion_reliability/
└── kfold/
    ├── glrx/
    │   ├── fold_00/
    │   │   ├── epoch_005.pth
    │   │   ├── epoch_010.pth
    │   │   └── best_model.pth
    │   └── ...
    └── ...
```

## Common Issues

### 1. "No checkpoint found"

Make sure you have trained SyntalNet checkpoints in the correct location:
```
checkpoints/_FINAL_80-20_NO-SOSEX_Stratified/kfold/fold_XX/
```

Check config `base_checkpoint_dir` points to the right location.

### 2. "CUDA out of memory"

Reduce batch size in the base SyntalNet config:
```yaml
training:
  batch_size: 4  # Reduce from 8
```

Or use a smaller device:
```bash
--device cuda:1
```

### 3. "No results found to aggregate"

Make sure you've run in `eval` or `both` mode, not just `train`.

## Customization

### Change Training Epochs

Edit the config file:
```yaml
reliability_training:
  num_epochs: 50  # Increase from 30
```

### Change Corruption Probability

```yaml
reliability_training:
  p_corrupt: 0.8  # Increase from 0.7
```

### Change Corruption Strength Range

```yaml
reliability_training:
  corruption_strengths:
    dropout: [0.2, 0.6]  # Stronger corruption
    noise: [0.2, 0.6]
    shuffle: [0.2, 0.6]
```

### Add Custom Corruption Sweep Points

```yaml
stress_test:
  corruption_configs:
    dropout: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Finer granularity
```

## Tips

1. **Start with one fold**: Test with `--fold 0` first to make sure everything works
2. **Monitor training**: Check logs in `logs/multimodal_fusion_reliability/kfold/VARIANT/fold_XX/`
3. **Use best checkpoint**: Best model is automatically saved as `best_model.pth`
4. **Parallel folds**: You can run different folds on different GPUs in parallel:
   ```bash
   python runner.py --config ... --fold 0 --device cuda:0 &
   python runner.py --config ... --fold 1 --device cuda:1 &
   ```

## Questions?

See the main README.md for detailed documentation.

