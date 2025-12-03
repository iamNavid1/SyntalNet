# Multimodal Reliability-Switch Experiments

This package hosts a streamlined reliability-switch experiment suite for SyntalNet.  
It rebuilds only the multimodal fusion head (GLR-X / UniformAvg / Concat+MLP) on
top of frozen, pretrained SyntalNet checkpoints and fine-tunes fusion + classifier
heads while simulating single-modality failures. Afterwards, it runs the same model
through systematic stress tests and (optionally) GLR-X allocation tracking.

## Highlights

- **Re-uses existing infrastructure**: datasets/loaders from `train.py`, optimizer +
  scheduler builders, loss logic via `experiments/ood_adaptation/finetuner.py`.
- **Minimal training surface**: only the multimodal fusion module (fresh init) and
  classification heads are trainable; encoders + branch mixers stay frozen.
- **Reliability-switch augmentation**: for every batch sample, exactly one modality
  is randomly corrupted (dropout/noise/shuffle mix) after branch pooling but before
  multimodal fusion. Validation stays clean.
- **Post-hoc stress tests**: evaluate robustness under controlled corruption sweeps
  (uniform and per-modality) using the canonical corruption functions from
  `experiments/multimodal_fusion/corruptions.py`.
- **Allocation tracking ready**: for GLR-X variants, we can reuse
  `AllocationTracker` to export gate statistics under escalating noise.

## Directory Layout

```
experiments/multimodal_reliability/
├── README.md
├── __init__.py
├── augmentations.py              # reliability-switch corruption logic
├── config.py                     # dataclasses + YAML loader
├── experiment.py                 # orchestration (training + evaluation)
├── model_utils.py                # model loading, fusion rebuild, freezing helpers
├── runner.py                     # CLI entry point
├── stress.py                     # clean/stress-test evaluation utilities
└── configs/
    ├── reliability_glrx.yaml
    ├── reliability_uniform_avg.yaml
    └── reliability_concat_mlp.yaml
```

## Configuration

Each YAML config is parsed into `ExperimentConfig` (see `config.py`). Key fields:

```yaml
base_config_path: "configs/SyntalNet.yaml"
base_checkpoint_dir: "checkpoints/_FINAL_RUN"
base_checkpoint_name: "epoch_030.pth"  # loaded per fold
cv_mode: "kfold"
n_folds: 5
output_root: "experiments/multimodal_reliability/results"

fusion:
  name: "glrx"                     # or "uniform_avg", "concat_mlp"
  kwargs:
    rank_pair: 16
    alloc_hidden: 32
    eps_floor: 0.03

reliability:
  p_corrupt: 0.7
  dropout_range: [0.2, 0.5]
  noise_range: [0.1, 0.4]
  shuffle_range: [0.2, 0.5]
  mix: {dropout: 0.4, noise: 0.4, shuffle: 0.2}

training:
  num_epochs: 30
  learning_rate: null              # defaults to base config LR
  weight_decay: null               # defaults to base config WD
  grad_clip: 1.0
  accum_steps: 1
  scheduler: "reduce_on_plateau"
  scheduler_factor: 0.5
  scheduler_patience: 3
  scheduler_threshold: 0.01
  scheduler_min_lr: 1.0e-7

stress:
  corruption_grid:
    dropout: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
    noise:   [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
    shuffle: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
  per_modality: true
  run_allocation: true
  allocation_noise_levels: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
```

## Training Workflow

1. **Load base model + checkpoint**  
   `model_utils.prepare_model` builds SyntalNet from `base_config_path`, loads the
   pretrained fold checkpoint, replaces `mm_fusion` with the requested variant, and
   wraps it with `ReliabilityFusionWrapper`, which injects the augmentation.

2. **Freeze backbone**  
   All branch encoders (including multi-channel mixers) are frozen. Only the new
   `mm_fusion` parameters and any classifier weights/biases stay trainable.
   A small hook keeps branches in evaluation mode even when the rest of the model
   is in `.train()` to avoid stochastic dropout in the frozen backbone.

3. **Fine-tuning loop**  
   `experiment.py` instantiates `FineTuner` (same as OOD adaptation) with the
   reliability loaders. Optimizer and scheduler are still built via
   `utils.optimizer` / `utils.scheduler`, now using ReduceLROnPlateau. The wrapper
   ensures that for every training batch:

   ```
   if bernoulli(p_corrupt):
       pick one modality → pick one corruption type
       apply (dropout / noise / shuffle) to that branch embedding
   ```

   Validation uses clean data (`fusion_wrapper.enable_corruption(False)`).

4. **Checkpointing**  
   Both the latest and best-by-validation-loss states are written under
   `results/.../fold_xx/checkpoints/`.

## Stress Tests & Allocation Tracking

After training (or when running evaluation-only mode), the runner:

1. Reloads the **last** checkpoint for that fold (`last.pth`) and disables the
   augmentation wrapper (no reliability-switch during evaluation).
2. Computes clean metrics on the validation split (no corruption).
3. Sweeps the configured corruption grids. For each corruption type/value and
   optionally per modality, we:
   - Re-encode frozen branches,
   - Apply the selected corruption via `experiments/multimodal_reliability/corruptions.py`,
   - Run fusion + classifiers, collect metrics.
4. Saves `clean_metrics.csv` and `stress_tests.csv` inside `results/fold_xx/`.
5. If `run_allocation` is enabled and the variant supports GLR-X gating, runs
   `AllocationTracker` with the provided noise levels and writes the CSV export
   next to the other results.

### CSV Outputs in Detail

Each `(variant, fold)` writes a small family of CSV/JSON files under
`results/fold_xx/`. This section documents **what each file contains, how rows
are constructed, and how to interpret the columns**.

### 1. `clean_metrics.csv`

**Where it comes from**:  
Produced by `ReliabilityStressTester.evaluate_clean()` and saved by
`ReliabilityExperimentRunner._evaluate_fold`.

**What it represents**:  
Performance of the trained model on **uncorrupted validation data**, per
classification split and head.

**Row structure**:

- One row per combination of:
  - `split` ∈ `{individual, group}`
  - `head` ∈ classifier head names, typically `Engagement`, `Valence`

**Columns**:

- **`split`**: classification split
  - `individual`: per-person predictions (person-temporal view).
  - `group`: group-level predictions (group-temporal view).
- **`head`**: name of the classification head (e.g. `Engagement`, `Valence`).
- **`accuracy`**: standard (micro) accuracy over all samples for this head.
- **`f1_macro`**: macro-averaged F1 score across classes for this head.
- **`auroc`**: macro-averaged AUROC across classes for this head.
- **`auprc`**: macro-averaged AUPRC across classes for this head.

**How rows are constructed**:

1. The evaluator iterates over the validation loader, collects **all logits** and
   **all targets** for each `(split, head)` pair.
2. At the end, it calls `compute_classification_metrics_from_logits` for each
   head, which computes F1-macro, AUROC-macro, and AUPRC-macro from logits +
   ground-truth labels.
3. A row is emitted for every `(split, head)` for which metrics exist.

**Typical usage**:

- Use this file as the **baseline reference** when judging robustness:
  - Compare stressed metrics at `(corruption_type, param=0.0)` and/or `(modality="all")`
    against the numbers here.
  - Sanity-check that training achieved reasonable clean performance before
    looking at degradation curves.

### 2. `stress_tests.csv`

**Where it comes from**:  
Produced by `ReliabilityStressTester.run_stress_tests`, which:

1. Calls `evaluate_clean()` once (baseline).
2. For each corruption type and sweep value (and optionally per modality), runs
   evaluation with the selected corruption.
3. Reuses the clean baseline for any parameter values that are effectively zero
   (no-op corruption) instead of recomputing.

**What it represents**:  
Performance degradation under corruption sweeps, for:

- Different **corruption types**: `dropout`, `noise`, `shuffle`, optionally `rescale`.
- Different **corruption strengths** given by the `stress.corruption_grid`.
- Both **uniform** corruption (`modality="all"`) and **per-modality**
  corruption (`modality` equal to a specific branch name).

**Row structure**:

- One row per combination of:
  - `corruption_type` ∈ keys of `stress.corruption_grid`
  - `corruption_param` ∈ the sweep values for that type
  - `modality` ∈ `{ "all" } ∪ branch_names`, where branch_names are typically
    `Videokinetic`, `Dialogue`, `Acoustic`
  - `split` ∈ `{individual, group}`
  - `head` ∈ classifier head names

**Columns**:

- **`corruption_type`**:
  - `dropout`: per-modality dropout on z-embeddings (see `corruptions.dropout_corruption`)
  - `noise`: additive Gaussian noise scaled to per-sample std
  - `shuffle`: sample-level shuffling within the batch for the selected modality
  - `rescale` (if used): feature magnitude rescaling
- **`corruption_param`**: value from the corresponding sweep list, e.g. `0.0`,
  `0.15`, `0.30`, etc. The exact semantics depend on `corruption_type`:
  - `dropout`: probability a sample is zeroed for that modality.
  - `noise`: relative noise strength (multiplicative factor on per-sample std).
  - `shuffle`: fraction of samples whose embedding is replaced by another sample.
  - `rescale`: multiplicative scaling factor for embedding norms.
- **`modality`**:
  - `"all"`: the corruption is applied identically to all branches (uniform case).
  - Branch name (e.g. `Videokinetic`, `Dialogue`, `Acoustic`): only that
    branch’s embedding is corrupted; other branches remain clean.
- **`split`**: `individual` or `group` (same meaning as in `clean_metrics.csv`).
- **`head`**: classifier head name (e.g. `Engagement`, `Valence`).
- **`accuracy`**: accuracy under this corruption setting.
- **`f1_macro`**: macro-averaged F1 score under this corruption setting.
- **`auroc`**: macro-averaged AUROC under this corruption setting.
- **`auprc`**: macro-averaged AUPRC under this corruption setting.

**How rows are constructed**:

For each `(corruption_type, values)` in `stress.corruption_grid`:

1. **Uniform corruption (`modality=None`)**:
   - For each `param` in `values`:
     - If a **baseline** dict was provided and `param == 0.0`, metrics are taken
       directly from the clean baseline (no extra forward passes).
     - Otherwise, a corruption function `fn = CORRUPTION_FUNCS[corruption_type]`
       is created, and the evaluator:
       - For each batch, re-encodes branches (`model.branches`), applies
         `fn(z_list, param, which_mod=None)`, then passes through `mm_fusion`
         and classifiers to accumulate logits/targets.
       - Computes metrics via `compute_classification_metrics_from_logits` per
         `(split, head)`.
     - A row is written for **each `(split, head)`** with these metrics, plus the
       corruption metadata fields (`corruption_type`, `corruption_param`,
       `modality="all"`).

2. **Per-modality corruption (`modality_idx = 0..M-1`)**:
   - For each modality index, the same loop runs, but the corruption function
     is called as `fn(z_list, param, which_mod=modality_idx)`, so only one
     branch is perturbed.
   - Again, for `param == 0.0` and when a baseline is available, we reuse the
     clean metrics. Rows are constructed with `modality` set to the branch name.

**Typical usage**:

- Plot **degradation curves**:
  - For a fixed `(split, head)` and `modality="all"`, plot `f1_macro` vs
    `corruption_param` for each `corruption_type`. This shows overall
    robustness to uniform corruption.
  - For `per_modality=true`, fix `(split, head, corruption_type)` and compare
    curves across `modality` to see which branch the model relies on most.
- Compare **variants**:
  - Aggregate across folds and plot GLR-X vs UniformAvg vs ConcatMLP on the same
    axes using the same CSV schema.

### 3. `allocation_tracking_*.csv` (GLR-X only, optional)

**Where it comes from**:  
Generated by `experiments.multimodal_fusion.allocation_tracker.run_allocation_tracking`
when `stress.run_allocation` is `true` and the fusion variant supports
allocation/gate extraction (GLR-X).

**What it represents**:  
Summary statistics of GLR-X **allocation weights** under controlled noise added
to each modality in turn.

**Row structure**:

- One row per combination of:
  - `noisy_modality`: index of the branch to which noise was applied.
  - `noise_level`: one entry from `stress.allocation_noise_levels`.
  - `target_modality`: index of the modality whose allocation we are measuring.

**Columns** (typical from `allocation_tracker.py`):

- **`variant`**: name of the fusion variant (e.g. `glrx`).
- **`fold`**: fold index.
- **`noisy_modality`**: index of the modality that received additional noise.
- **`noise_level`**: the scalar noise-level parameter used for this run.
- **`target_modality`**: index of the modality whose allocation stats are recorded.
- **`allocation_mean`**: mean allocation weight assigned to `target_modality`
  across all samples for this `(noisy_modality, noise_level)` combination.
- **`allocation_std`**: standard deviation of allocation weights across samples.
- **`n_samples`**: total number of samples aggregated into these statistics.

**How rows are constructed**:

1. For each `noisy_modality` and `noise_level`, the tracker:
   - Runs the model on the validation loader, injecting Gaussian noise into the
     selected branch’s embedding only.
   - Calls `mm_fusion.get_aux()` to retrieve `aux["alloc"]` (B × M allocation
     matrix) per batch, and concatenates across all batches.
2. It then computes, for each `target_modality`, the mean and std of the
   corresponding allocation column and emits one row per modality.

**Typical usage**:

- Inspect **allocation shifts** as one modality is degraded:
  - Expectation: when you add noise to modality j, allocation should gradually
    move away from j and towards its more reliable peers.
  - Compare GLR-X’s allocation behaviour under noise to that of baselines that
    have no adaptive gating.

### 4. `training_history.json`

**Where it comes from**:  
Written by `ReliabilityExperimentRunner.train_fold` after training when it was
run in a mode that includes training (`--mode train` or `--mode both`).

**What it represents**:  
Per-epoch summary of training and (optional) validation loss.

**Structure**:

- A JSON list of objects, each with:
  - `epoch`: 1-based epoch index.
  - `train_loss`: average training loss for that epoch (over all steps).
  - `val_loss`: validation loss for that epoch, or `null` if validation was
    skipped (e.g. `training.val_interval <= 0`).

**Typical usage**:

- Plot **training curves**:
  - Quickly peek at convergence and overfitting behaviour.
  - Cross-check that fine-tuning actually reduced loss before looking at stress
    tests.

## Usage

Run the new experiment runner with your preferred config:

```bash
python experiments/multimodal_reliability/runner.py \
    --config experiments/multimodal_reliability/configs/reliability_glrx.yaml \
    --mode both \
    --fold 0
```

Options:
- `--mode train|eval|both`: train only, evaluate existing checkpoints, or do both.
- `--fold <idx>`: restrict to a single fold (defaults to all `n_folds`).
- `--device cuda:0|cpu`: override device without editing the YAML.

Outputs live under `output_root/cv_mode/<variant>/fold_xx/`:

```
fold_00/
├── checkpoints/{best,last}.pth
├── logs/run.log
└── results/
    ├── clean_metrics.csv
    ├── stress_tests.csv
    ├── allocation_tracking_glrx.csv   # optional
    └── training_history.json          # only when training ran
```

These CSV files can be ingested directly into notebooks or plotting scripts to
compare GLR-X against UniformAvg and ConcatMLP on the same reliability-switch
protocol.

