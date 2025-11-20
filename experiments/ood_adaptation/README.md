# OOD Adaptation Experiments

Transfer learning experiments for out-of-distribution (OOD) adaptation using Leave-One-Group-Out (LOGO) cross-validation. Evaluates how well models adapt to new groups with limited fine-tuning data.

## Overview

This experiment tests model adaptation to held-out groups by:
- **Experiment 1**: Data-portion sweep - Tests performance with varying amounts of fine-tuning data (0% to 20%)
- **Experiment 2**: Epoch sweep - Tests performance with varying numbers of fine-tuning epochs

Models are fine-tuned with frozen encoders, training only the classifier heads on stratified samples from the held-out group.

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
- `--exp2-only`: Run only Experiment 2 (epoch sweep)
- `--data-proportions`: Data proportions for Experiment 1 (default: `[0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]`)
- `--finetune-epochs`: Number of fine-tuning epochs for Experiment 1 (default: `10`)
- `--fixed-proportion`: Fixed data proportion for Experiment 2 (default: `0.10`)
- `--epochs-to-extract`: Epochs to extract from Exp1 for Exp2 (default: `[2, 4, 6, 8, 10]`)
- `--extra-epochs`: Extra epochs to train for Exp2 (default: `[12, 15, 18, 20]`)
- `--lr-divisor`: Divide base LR by this factor for fine-tuning (default: `30`)
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

- `ood_adaptation_exp1.csv`: Experiment 1 results (data-portion sweep)
- `ood_adaptation_exp2.csv`: Experiment 2 results (epoch sweep)
- `ood_adaptation.log`: Execution log

## Experiment Details

### Experiment 1: Data-Portion Sweep
- For each LOGO fold (held-out group):
  - Load base model checkpoint
  - For each data proportion (0%, 1%, 2%, 5%, 10%, 15%, 20%):
    - Sample stratified subset from held-out group for fine-tuning
    - Use remainder as test set
    - Fine-tune for fixed number of epochs (default: 10)
    - Evaluate after each epoch

### Experiment 2: Epoch Sweep
- Uses results from Experiment 1
- For each LOGO fold:
  - Load checkpoints from Experiment 1 at specified epochs
  - Continue training for additional epochs
  - Evaluates performance across extended training

## Fine-Tuning Strategy

- **Frozen encoders**: Only classifier heads are trained
- **Stratified sampling**: Ensures class balance in fine-tuning data
- **Reduced learning rate**: Base LR divided by `--lr-divisor` (default: 30)
- **Zero-shot baseline**: Proportion 0.0 provides baseline without fine-tuning

