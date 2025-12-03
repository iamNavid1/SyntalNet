# OOD Adaptation Results Visualization

This script visualizes the results from three OOD (Out-of-Distribution) adaptation experiments.

## Overview

The visualization creates a figure with **3 horizontal panels**, each representing one experiment:

### Panel 1: Experiment 1 - Data-Portion Sweep
- **Purpose**: Shows how model performance improves with increasing amounts of fine-tuning data
- **X-axis**: 5 construct clusters (Engagement, Lead, Synchrony, Confidence, Transition)
- **Y-axis**: AUPRC (macro) scores
- **Bars**: 6 bars per cluster
  - Zero-shot (0% training data, epoch 0)
  - 5%, 10%, 15%, 20%, 25% training data proportions (all at epoch 20, the final epoch)
- **Error bars**: Standard deviation across 10 LOGO folds
- **Note**: All non-zero proportions show results from epoch 20 (the last epoch of fine-tuning)

### Panel 2: Experiment 2 - Epoch Sweep
- **Purpose**: Shows how performance evolves during fine-tuning across different data proportions
- **X-axis**: Epoch numbers [0 (zero-shot), 2, 8, 12, 16, 20]
- **Y-axis**: AUPRC (macro) averaged over all 5 constructs
- **Lines**: 5 colored lines, one for each training data proportion (5%, 10%, 15%, 20%, 25%)
- **Point**: Zero-shot baseline (epoch 0)

### Panel 3: Experiment 3 - Frozen Backbone with Item Split
- **Purpose**: Compares zero-shot performance vs. frozen backbone fine-tuning (80/20 split)
- **X-axis**: 5 construct clusters (Engagement, Lead, Synchrony, Confidence, Transition)
- **Y-axis**: AUPRC (macro) scores
- **Bars**: 2 bars per cluster
  - Zero-shot (baseline)
  - Frozen Backbone with 80% training data (last epoch)
- **Error bars**: Standard deviation across 10 LOGO folds

## Usage

### Basic Usage

```bash
python experiments/viz/OOD_adaptation_results.py \
    --input-dir experiments/ood_adaptation_results \
    --output-dir experiments/viz_results/ood_adaptation
```

### Command-Line Arguments

- `--input-dir`: Directory containing experiment CSV files (default: `./experiments/ood_adaptation_results`)
  - Required files:
    - `exp1_data_portion_sweep_aggregated.csv`
    - `exp2_epoch_sweep_aggregated.csv`
    - `exp3_frozen_backbone_item_split_aggregated.csv`

- `--output-dir`: Directory to save output figures (default: `./experiments/viz_results/ood_adaptation`)
  - Creates two files:
    - `{figure_name}.png` (high-resolution raster image)
    - `{figure_name}.pdf` (vector graphics)

- `--figure-name`: Base name for output files (default: `ood_adaptation_results`)

- `--no-show`: Don't display the plot, only save to files

### Examples

**Generate figures without displaying:**
```bash
python experiments/viz/OOD_adaptation_results.py \
    --input-dir experiments/ood_adaptation_results \
    --output-dir experiments/viz_results/ood_adaptation \
    --no-show
```

**Custom figure name:**
```bash
python experiments/viz/OOD_adaptation_results.py \
    --input-dir experiments/ood_adaptation_results \
    --figure-name my_ood_results
```

## Input Data Format

The script expects aggregated CSV files with the following structure:

### Experiment 1 CSV
- Columns: `proportion`, `epoch`, `n_folds`, `{construct}_{metric}_mean`, `{construct}_{metric}_std`, ...
- Proportions: 0.0, 0.05, 0.10, 0.15, 0.20, 0.25
- Constructs: `individual_Engagement`, `individual_Lead`, `group_Synchrony`, `group_Confidence`, `group_Transition`

### Experiment 2 CSV
- Similar structure to Experiment 1
- Proportions: 0.05, 0.10, 0.15, 0.20, 0.25
- Epochs: 2, 8, 12, 16, 20

### Experiment 3 CSV
- Similar structure to Experiment 1
- Proportion: 0.8 (80% train, 20% test)
- Epochs: 1-30 (script uses last epoch)

## Output

The script generates two files in the output directory:

1. **PNG file**: High-resolution (350 DPI) raster image suitable for presentations and web
2. **PDF file**: Vector graphics suitable for publication and printing

### Figure Specifications

- **Figure size**: 21 × 5.5 inches (3 panels side-by-side)
- **Resolution**: 350 DPI
- **Background**: Light gray (#FAFAFB)
- **Color palette**: 
  - Zero-shot: Dark gray (#2C2C2C)
  - 5%: Blue (#5B8FF9)
  - 10%: Green (#5AD8A6)
  - 15%: Purple-gray (#5D7092)
  - 20%: Yellow-orange (#F6BD16)
  - 25%: Red-orange (#E8684A)

## Constructs

The visualization includes 5 constructs (in display order):

1. **Engagement** (Individual label)
2. **Lead** (Individual label)
3. **Synchrony** (Group label)
4. **Confidence** (Group label)
5. **Transition** (Group label)

## Dependencies

- Python 3.7+
- numpy
- pandas
- matplotlib

## Related Scripts

- `experiments/ood_adaptation/runner.py`: Script that generates the experiment data
- `experiments/viz/SoSEX_ablation_result.py`: Similar visualization style for ablation studies
- `experiments/viz/BSCX_ablation_results.py`: BSCX ablation visualization
- `experiments/viz/GLRX_ablation_results.py`: GLRX ablation visualization

## Notes

- All error bars represent standard deviation across 10 LOGO (Leave-One-Group-Out) folds
- Panel 1 (Experiment 1) uses **epoch 20** (the final epoch) for all non-zero proportions, and **epoch 0** for zero-shot
- Panel 2 shows the **average AUPRC** across all 5 constructs to reduce clutter
- Panel 3 uses the **last epoch** from Experiment 3 (typically epoch 30)
- The script automatically handles missing data (shows NaN if data is not available)

## Troubleshooting

**Error: "CSV file not found"**
- Ensure the input directory contains all three required CSV files
- Check that the file names match exactly (case-sensitive)

**Error: "No module named 'matplotlib'"**
- Install required dependencies: `pip install numpy pandas matplotlib`

**Empty plots or missing data**
- Check that the CSV files contain data for all expected proportions and constructs
- Verify that the column names match the expected format (`{construct}_auprc_macro_mean`, etc.)

