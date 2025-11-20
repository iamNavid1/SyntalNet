# OOD Adaptation Experiments

Transfer learning experiments for out-of-distribution (OOD) adaptation using Leave-One-Group-Out (LOGO) cross-validation. Evaluates how well models adapt to new groups with limited fine-tuning data.

## Overview

This experiment tests model adaptation to held-out groups through three experiments:
- **Experiment 1**: Data-portion sweep - Tests performance with varying amounts of fine-tuning data (0% to 25%), with all layers unfrozen
- **Experiment 2**: Epoch sweep analysis - Post-processes Experiment 1 results to extract metrics at specific epochs
- **Experiment 3**: Frozen backbone with item-mode split - Tests transfer learning with frozen backbone using standard 80/20 train/test split

## Usage

```bash
python experiments/ood_adaptation/runner.py \
    --config configs/SyntalNet_logo.yaml \
    --checkpoint-dir checkpoints/SyntalNet_logo \
    --output-dir experiments/ood_adaptation_results/
```

### Arguments

- `--config`: Path to YAML config file used for base LOGO training
- `--checkpoint-dir`: Base directory containing LOGO checkpoints (e.g., `checkpoints/SyntalNet`)
- `--output-dir`: Directory to save experiment results (default: `results/ood_adaptation`)
- `--exp1-only`: Run only Experiment 1 (data-portion sweep)
- `--exp2-only`: Run only Experiment 2 (epoch sweep analysis)
- `--exp3-only`: Run only Experiment 3 (frozen backbone with item split)
- `--data-proportions`: Data proportions for Experiment 1 (default: `[0.0, 0.05, 0.10, 0.15, 0.20, 0.25]`)
- `--full-finetune-epochs`: Number of epochs for full fine-tuning (all layers) in Experiment 1 (default: `20`)
- `--epochs-to-extract`: Epochs to extract from Exp1 for Exp2 (default: `[2, 8, 12, 16, 20]`)
- `--partial-finetune-epochs`: Number of epochs for partial fine-tuning (frozen backbone) in Experiment 3 (default: `10`)
- `--exp3-train-ratio`: Train ratio for Experiment 3 item-mode split (default: `0.8`)
- `--lr-divisor`: Divide base LR by this factor for Experiment 1 fine-tuning (default: `5`)
- `--seed`: Random seed for data sampling (default: `42`)
- `--device`: Device to use (default: auto-detect)

## Checkpoint Directory Structure

The experiment expects LOGO checkpoints organized as:

```
checkpoint_dir/
├── logo/
│   ├── fold_00/
│   │   ├── best.pth (or latest.pth, epoch_100.pth, epoch_*.pth)
│   ├── fold_01/
│   │   ├── best.pth
│   └── ...
```

## Output

- `exp1_data_portion_sweep_per_epoch.csv`: Experiment 1 per-epoch results
- `exp1_data_portion_sweep_aggregated.csv`: Experiment 1 aggregated results
- `exp1_data_portion_sweep.json`: Experiment 1 raw results (JSON)
- `exp2_epoch_sweep_per_epoch.csv`: Experiment 2 per-epoch results
- `exp2_epoch_sweep_aggregated.csv`: Experiment 2 aggregated results
- `exp2_epoch_sweep.json`: Experiment 2 raw results (JSON)
- `exp3_frozen_backbone_item_split_per_epoch.csv`: Experiment 3 per-epoch results
- `exp3_frozen_backbone_item_split_aggregated.csv`: Experiment 3 aggregated results
- `exp3_frozen_backbone_item_split.json`: Experiment 3 raw results (JSON)
- `ood_adaptation.log`: Execution log

## Experiment Details

### Experiment 1: Data-Portion Sweep

For each LOGO fold (held-out group):
- Load base model checkpoint
- For each data proportion (0%, 5%, 10%, 15%, 20%, 25%):
  - Sample stratified subset from held-out group for fine-tuning
  - Use remainder as test set
  - **Unfreeze all layers** (encoder/backbone + classifier)
  - Fine-tune for **20 epochs** (default)
  - Learning rate: `base_lr / 5`
  - LR schedule: **Warmup only, then constant LR** (no decay)
  - Evaluate after each epoch

**Key Settings:**
- All layers trainable (encoder + classifier)
- Learning rate: `base_lr / 5`
- Scheduler: `warmup_constant` (warmup then constant)
- Epochs: 20
- Zero-shot baseline: Proportion 0.0 provides baseline without fine-tuning

### Experiment 2: Epoch Sweep Analysis

**Post-processing step** (no new training):
- Uses logged metrics from Experiment 1
- For each LOGO fold and each data portion used in Experiment 1:
  - Extracts metrics at epochs **[2, 8, 12, 16, 20]**
  - Organizes into separate results structure/file
- No model training is performed

**Key Settings:**
- Post-processing only
- Extracts epochs: [2, 8, 12, 16, 20]
- Works for all folds and all proportions from Experiment 1

### Experiment 3: Frozen Backbone with Item-Mode Split

For each LOGO fold:
- Load base model checkpoint
- Take complete left-out group for that fold
- **Data split**: Use existing "item" mode splitting logic to create 80/20 train/test split
  - 80% of items → fine-tuning train set
  - 20% of items → test set
- **Freeze entire backbone/encoder**
- **Train only classifiers**
- Learning rate: **same as base training** (`base_lr`)
- LR schedule: **Warmup only, then constant LR** (no decay)
- Fine-tune for **10 epochs**
- Evaluate after each epoch

**Key Settings:**
- Frozen backbone (only classifiers trainable)
- Item-mode 80/20 split
- Learning rate: `base_lr` (same as base training)
- Scheduler: `warmup_constant` (warmup then constant)
- Epochs: 10

## Fine-Tuning Strategies

### Experiment 1
- **Unfrozen encoders**: All layers (encoder + classifier) are trained
- **Stratified sampling**: Ensures class balance in fine-tuning data
- **Reduced learning rate**: Base LR divided by 5
- **Warmup-constant scheduler**: Warmup phase, then constant LR

### Experiment 2
- **Post-processing**: No training, extracts results from Experiment 1 logs

### Experiment 3
- **Frozen encoders**: Only classifier heads are trained
- **Item-mode split**: Standard 80/20 train/test split at item level
- **Base learning rate**: Same LR as base training
- **Warmup-constant scheduler**: Warmup phase, then constant LR

