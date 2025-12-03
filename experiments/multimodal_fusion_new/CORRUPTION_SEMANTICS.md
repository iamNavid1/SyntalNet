# Corruption Semantics: Implementation Details

This document describes the corruption semantics implemented in the reliability-switch experiments, explaining the specific behavior of each corruption type.

## Overview

The `ReliabilitySwitchCorruptor` applies three types of corruption to branch embeddings:
1. **Dropout**: Modality-level scaling
2. **Noise**: Per-sample Gaussian noise
3. **Shuffle**: Cross-sample Bernoulli mixing

Each corruption has specific semantics that differ from typical augmentation strategies.

---

## 1. Dropout Corruption

### Semantics
**Modality-level scaling** rather than element-wise dropout.

### Implementation
```python
z_corrupted = z * (1.0 - dropout_scale)
```

### Key Differences from Standard Dropout
| Standard Dropout | Reliability-Switch Dropout |
|------------------|----------------------------|
| Element-wise Bernoulli mask | Whole-embedding scaling |
| Randomly zeros individual features | Uniformly scales all features |
| Needs scaling factor `1/(1-p)` | Direct scaling by `(1-p)` |
| Breaks feature relationships | Preserves relative feature magnitudes |

### Rationale
- Simulates a **modality being less reliable** rather than missing features
- Maintains the **relative structure** of the embedding
- Models scenarios where the entire modality is **degraded uniformly** (e.g., low audio volume, poor video quality)

### Example
```python
z = torch.tensor([[1.0, 2.0, 3.0]])  # Original embedding
dropout_scale = 0.7

# Standard dropout (element-wise, p=0.7)
mask = torch.bernoulli(torch.full_like(z, 0.3))
z_standard = z * mask / 0.3
# Result: [[3.33, 0.0, 0.0]] (random)

# Reliability-switch dropout (modality-level)
z_reliability = z * (1.0 - dropout_scale)
# Result: [[0.3, 0.6, 0.9]] (deterministic scaling)
```

---

## 2. Noise Corruption

### Semantics
**Per-sample Gaussian noise** scaled by sample-specific statistics.

### Implementation
```python
# Compute per-sample statistics
mean_sample = z.mean(dim=1, keepdim=True)
std_sample = ((z - mean_sample) ** 2).mean(dim=1, keepdim=True).sqrt()

# Add noise scaled by sample std
epsilon = torch.randn_like(z)
z_corrupted = z + noise_level * std_sample * epsilon
```

### Key Differences from Standard Noise
| Standard Noise | Reliability-Switch Noise |
|----------------|--------------------------|
| Batch-level or global scale | Per-sample scale |
| `z + σ_global * ε` | `z + α * σ_sample * ε` |
| Same noise magnitude for all samples | Adaptive to sample magnitude |
| Can overwhelm small-magnitude samples | Preserves sample-relative SNR |

### Rationale
- **Adaptive to sample characteristics**: Samples with larger magnitude get proportionally larger noise
- **Preserves signal-to-noise ratio**: Noise level is relative to the sample's own scale
- Models **measurement noise** that scales with signal strength (realistic sensor behavior)

### Example
```python
# Sample 1: Large magnitude
z1 = torch.tensor([[10.0, 20.0, 30.0]])
std1 = z1.std()  # ~8.16

# Sample 2: Small magnitude  
z2 = torch.tensor([[0.1, 0.2, 0.3]])
std2 = z2.std()  # ~0.0816

noise_level = 0.35

# Reliability-switch noise (per-sample)
noise1 = noise_level * std1 * torch.randn_like(z1)  # ~2.86 magnitude
noise2 = noise_level * std2 * torch.randn_like(z2)  # ~0.0286 magnitude

# Result: Noise magnitude proportional to sample magnitude
```

---

## 3. Shuffle Corruption

### Semantics
**Cross-sample contradictions** via Bernoulli mixing with different batch samples.

### Implementation
```python
# Select different sample from batch
perm = torch.randint(0, B, (n_samples,), device=device)

# Avoid trivial permutation (same sample)
same = (perm == sample_indices)
if same.any():
    perm[same] = (perm[same] + 1) % B

# Bernoulli mask for partial mixing
mask = (torch.rand_like(z) < shuffle_fraction).float()

# Mix embeddings
z_corrupted = mask * z[perm] + (1.0 - mask) * z
```

### Key Differences from Standard Shuffle
| Standard Shuffle | Reliability-Switch Shuffle |
|------------------|----------------------------|
| Permute within batch | Mix between samples |
| All samples shuffled | Only selected samples |
| Full replacement | Partial Bernoulli mixing |
| No cross-sample information | Creates contradictions |

### Rationale
- **Creates semantic contradictions**: Mixes incompatible information from different samples
- **Partial mixing** (not full replacement): More realistic than complete substitution
- Models **cross-talk** or **interference** between modalities (e.g., audio from different speakers)

### Example
```python
# Sample embeddings
z = torch.tensor([
    [1.0, 2.0, 3.0],  # Sample 0
    [4.0, 5.0, 6.0],  # Sample 1
    [7.0, 8.0, 9.0],  # Sample 2
])

# For sample 0, mix with sample 2
perm = [2]  # Select sample 2
shuffle_fraction = 0.5

# Bernoulli mask (50% chance per element)
mask = torch.tensor([[1.0, 0.0, 1.0]])  # Random

# Result for sample 0
z_corrupted = mask * z[2] + (1.0 - mask) * z[0]
# = [1.0, 0.0, 1.0] * [7.0, 8.0, 9.0] + [0.0, 1.0, 0.0] * [1.0, 2.0, 3.0]
# = [7.0, 2.0, 9.0]  (mix of sample 0 and sample 2)
```

---

## Configuration

### ReliabilitySwitchConfig

```python
@dataclass
class ReliabilitySwitchConfig:
    p_corrupt: float = 0.7          # Probability of corrupting a sample
    
    # Corruption type mixture (normalized to sum to 1.0)
    mix_dropout: float = 0.4        # 40% dropout
    mix_noise: float = 0.4          # 40% noise
    mix_shuffle: float = 0.2        # 20% shuffle
    
    # Corruption parameters
    dropout_scale: float = 0.7      # Scale factor in [0, 1]
    noise_level: float = 0.35       # Multiplier for sample std
    shuffle_fraction: float = 0.5   # Bernoulli mask probability
```

### YAML Configuration

```yaml
reliability_training:
  p_corrupt: 0.7
  
  corruption_mix:
    dropout: 0.4
    noise: 0.4
    shuffle: 0.2
  
  dropout_scale: 0.7
  noise_level: 0.35
  shuffle_fraction: 0.5
```

---

## Stress Testing

During stress tests, the same corruption semantics are applied:

```python
# Uniform corruption (all modalities)
z_corrupted = apply_uniform_corruption(
    z_list=[z_video, z_audio, z_text],
    corruption_type="dropout",  # or "noise", "shuffle"
    corruption_param=0.5
)

# Single-modality corruption
z_corrupted = apply_single_modality_corruption(
    z_list=[z_video, z_audio, z_text],
    modality_idx=0,  # Corrupt video only
    corruption_type="noise",
    corruption_param=0.35
)
```

---

## Design Rationale

### Why These Semantics?

1. **Modality-level Dropout**
   - Real-world: Entire sensors degrade, not individual features
   - Preserves embedding structure
   - Easier for fusion to detect and down-weight

2. **Per-sample Noise**
   - Real-world: Noise scales with signal
   - Fairer across diverse samples
   - Avoids overwhelming small-magnitude embeddings

3. **Cross-sample Shuffle**
   - Real-world: Interference and cross-talk
   - Creates semantic contradictions
   - Tests fusion's ability to detect inconsistencies

### Comparison to Standard Augmentation

| Aspect | Standard Augmentation | Reliability-Switch |
|--------|----------------------|-------------------|
| Goal | Increase data diversity | Test robustness to corruption |
| Scope | Element-wise | Modality-level |
| Statistics | Global/batch | Per-sample adaptive |
| Mixing | Within sample | Cross-sample |

---

## Usage Example

```python
from experiments.multimodal_fusion_new.corruptions import (
    ReliabilitySwitchConfig,
    ReliabilitySwitchCorruptor
)

# Create config
config = ReliabilitySwitchConfig(
    p_corrupt=0.7,
    mix_dropout=0.4,
    mix_noise=0.4,
    mix_shuffle=0.2,
    dropout_scale=0.7,
    noise_level=0.35,
    shuffle_fraction=0.5,
)

# Create corruptor
corruptor = ReliabilitySwitchCorruptor(config)

# Apply corruption (in training loop)
z_branch = [z_video, z_audio, z_text]  # List of (B, D) tensors
z_corrupted = corruptor(z_branch)

# For each sample:
# - 70% chance of corruption
# - If corrupted, picks ONE modality and ONE corruption type
# - Corruption type sampled from mixture (40/40/20)
# - Applied with semantics described above
```

---

## References

For more details, see:
- `experiments/multimodal_fusion_new/corruptions.py` - Implementation
- `experiments/multimodal_fusion_new/README.md` - Usage guide
- `experiments/multimodal_fusion_new/EXPERIMENT_SUMMARY.md` - Experimental design

