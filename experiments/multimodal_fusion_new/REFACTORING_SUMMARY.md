# Refactoring Summary: Corruption Utilities

## Overview

The corruption utilities have been refactored to implement a class-based `ReliabilitySwitchCorruptor` with improved semantics matching the reference implementation. The key changes focus on:

1. **Class-based design** with dataclass configuration
2. **Improved corruption semantics** (modality-level dropout, per-sample noise, cross-sample shuffle)
3. **Probabilistic corruption type mixing** instead of random uniform selection
4. **Backward compatibility** for stress testing functions

---

## What Changed

### 1. New Classes and Configs

#### `ReliabilitySwitchConfig` (dataclass)
```python
@dataclass
class ReliabilitySwitchConfig:
    p_corrupt: float = 0.7           # Probability of corrupting a sample
    
    # Corruption type mixture
    mix_dropout: float = 0.4         # 40%
    mix_noise: float = 0.4           # 40%
    mix_shuffle: float = 0.2         # 20%
    
    # Corruption parameters
    dropout_scale: float = 0.7       # Modality-level scale
    noise_level: float = 0.35        # Per-sample std multiplier
    shuffle_fraction: float = 0.5    # Bernoulli mask probability
```

**Features:**
- Clean, type-safe configuration
- Can be created from dict/YAML via `from_dict()` classmethod
- All parameters in one place

#### `ReliabilitySwitchCorruptor` (class)
```python
class ReliabilitySwitchCorruptor:
    def __init__(self, cfg: ReliabilitySwitchConfig):
        self.cfg = cfg
        # Normalize corruption mixture to probabilities
        self.mix = normalize([cfg.mix_dropout, cfg.mix_noise, cfg.mix_shuffle])
    
    def __call__(self, z_list: List[torch.Tensor]) -> List[torch.Tensor]:
        # Apply corruption with improved semantics
        ...
```

**Features:**
- Callable class (clean API: `corruptor(z_list)`)
- Maintains state (config, normalized mixture)
- Vectorized implementation for efficiency

### 2. Corruption Semantics Changes

#### Dropout: Element-wise → **Modality-level Scaling**

**Before:**
```python
mask = torch.bernoulli(torch.full_like(z, 1.0 - p))
z_corrupted = z * mask / (1.0 - p + 1e-8)  # Scaled dropout
```

**After:**
```python
scale = 1.0 - dropout_scale
z_corrupted = z * scale  # Whole embedding scaled
```

**Impact:**
- Preserves embedding structure
- Models uniform modality degradation
- More realistic for reliability testing

#### Noise: Batch-level → **Per-sample Statistics**

**Before:**
```python
# Compute per-feature std across batch
z_std = z.std(dim=0, keepdim=True).clamp_min(1e-6)
noise = torch.randn_like(z) * (scale * z_std)
```

**After:**
```python
# Compute per-sample mean and std
mean = z.mean(dim=1, keepdim=True)
std = ((z - mean) ** 2).mean(dim=1, keepdim=True).sqrt()
noise = torch.randn_like(z) * (noise_level * std)
```

**Impact:**
- Adaptive to sample magnitude
- Preserves sample-relative SNR
- Fairer across diverse samples

#### Shuffle: Per-sample permute → **Cross-sample Bernoulli Mixing**

**Before:**
```python
# Shuffle samples with probability p (per sample)
shuffle_mask = torch.rand(B, device=z.device) < p
if shuffle_mask.any():
    perm_indices = torch.randperm(num_shuffle, device=z.device)
    z_corrupted[shuffle_mask] = z[permuted_indices]
```

**After:**
```python
# Mix with different sample via Bernoulli mask
perm = torch.randint(0, B, (n_samples,), device=device)
perm_vals = z[perm]
mask = (torch.rand_like(z) < shuffle_fraction).float()
z_corrupted = mask * perm_vals + (1.0 - mask) * z
```

**Impact:**
- Creates semantic contradictions
- Partial mixing (not full replacement)
- Models cross-talk/interference

### 3. Corruption Type Selection

**Before:**
```python
# Uniform random selection from 3 types
corruption_types = torch.randint(0, 3, (num_corrupt,), device=device)
```

**After:**
```python
# Probabilistic sampling from mixture
mix = torch.tensor([mix_dropout, mix_noise, mix_shuffle])
mix = mix / mix.sum()  # Normalize
corruption_types = torch.multinomial(mix, num_corrupt, replacement=True)
```

**Impact:**
- Configurable corruption type distribution
- More control over augmentation strategy
- Can emphasize certain corruption types

### 4. API Changes

#### `ReliabilityTrainer` Constructor

**Before:**
```python
trainer = ReliabilityTrainer(
    model=model,
    ...
    p_corrupt=0.7,
    corruption_strengths={
        'dropout': (0.1, 0.5),
        'noise': (0.1, 0.5),
        'shuffle': (0.1, 0.5),
    },
    ...
)
```

**After:**
```python
config = ReliabilitySwitchConfig(
    p_corrupt=0.7,
    mix_dropout=0.4, mix_noise=0.4, mix_shuffle=0.2,
    dropout_scale=0.7, noise_level=0.35, shuffle_fraction=0.5
)
trainer = ReliabilityTrainer(
    model=model,
    ...
    corruption_config=config,
    ...
)
```

**Impact:**
- Type-safe configuration
- Clearer parameter meaning
- Easier to extend

#### YAML Configuration

**Before:**
```yaml
reliability_training:
  p_corrupt: 0.7
  corruption_strengths:
    dropout: [0.1, 0.5]
    noise: [0.1, 0.5]
    shuffle: [0.1, 0.5]
```

**After:**
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

**Impact:**
- Explicit parameters (not ranges)
- Corruption type distribution visible
- Self-documenting

### 5. Backward Compatibility

#### Legacy Function (DEPRECATED)

The old functional API is maintained for compatibility:

```python
def apply_reliability_switch_corruption(
    z_list: List[torch.Tensor],
    p_corrupt: float = 0.7,
    corruption_strengths: Optional[dict] = None,
) -> List[torch.Tensor]:
    """DEPRECATED: Use ReliabilitySwitchCorruptor class instead."""
    # Creates config and uses class internally
    cfg = ReliabilitySwitchConfig(
        p_corrupt=p_corrupt,
        dropout_scale=avg(corruption_strengths['dropout']),
        ...
    )
    corruptor = ReliabilitySwitchCorruptor(cfg)
    return corruptor(z_list)
```

#### Stress Test Functions

Updated to use new semantics:

```python
def apply_uniform_corruption(z_list, corruption_type, corruption_param):
    # Now uses:
    # - modality-level dropout
    # - per-sample noise
    # - cross-sample shuffle
    ...

def apply_single_modality_corruption(...):
    # Same semantics as uniform corruption
    ...
```

---

## Files Modified

### Core Implementation
- ✅ `corruptions.py` - Complete rewrite with new classes
- ✅ `reliability_trainer.py` - Updated to use new corruptor
- ✅ `runner.py` - Updated to create and pass config

### Configuration
- ✅ `configs/reliability_glrx.yaml` - New corruption config structure
- ✅ `configs/reliability_uniform_avg.yaml` - New corruption config structure
- ✅ `configs/reliability_concat_mlp.yaml` - New corruption config structure

### Documentation
- ✅ `README.md` - Updated corruption strategy section
- ✅ `EXPERIMENT_SUMMARY.md` - Updated with new semantics
- ✅ `CORRUPTION_SEMANTICS.md` - NEW: Detailed explanation
- ✅ `REFACTORING_SUMMARY.md` - NEW: This document

---

## Migration Guide

### For Users

If you have existing YAML configs, update them:

```yaml
# OLD
reliability_training:
  corruption_strengths:
    dropout: [0.1, 0.5]
    noise: [0.1, 0.5]
    shuffle: [0.1, 0.5]

# NEW
reliability_training:
  corruption_mix:
    dropout: 0.4
    noise: 0.4
    shuffle: 0.2
  dropout_scale: 0.7
  noise_level: 0.35
  shuffle_fraction: 0.5
```

### For Developers

If you were using the functional API:

```python
# OLD (still works but deprecated)
z_corrupted = apply_reliability_switch_corruption(
    z_list, p_corrupt=0.7, 
    corruption_strengths={'dropout': (0.1, 0.5), ...}
)

# NEW (recommended)
config = ReliabilitySwitchConfig(p_corrupt=0.7, ...)
corruptor = ReliabilitySwitchCorruptor(config)
z_corrupted = corruptor(z_list)
```

---

## Testing

### What to Test

1. **Corruption semantics**:
   - Dropout scales entire embedding
   - Noise uses per-sample statistics
   - Shuffle mixes cross-sample

2. **Corruption type selection**:
   - Respects mixture probabilities
   - Can configure distribution

3. **Backward compatibility**:
   - Old functional API still works
   - Stress test functions work

4. **Config loading**:
   - YAML configs load correctly
   - Config validates parameters

### Example Test

```python
def test_corruption_semantics():
    config = ReliabilitySwitchConfig(
        p_corrupt=1.0,  # Always corrupt
        mix_dropout=1.0,  # Only dropout
        dropout_scale=0.5,
    )
    corruptor = ReliabilitySwitchCorruptor(config)
    
    z = torch.ones(8, 128)  # Batch of 8, dim 128
    z_corrupted = corruptor([z])
    
    # Should be scaled by (1 - 0.5) = 0.5
    assert torch.allclose(z_corrupted[0], z * 0.5)
```

---

## Benefits

### Improved Semantics
- ✅ More realistic corruption (modality-level, per-sample)
- ✅ Better models real-world reliability issues
- ✅ Easier to interpret results

### Better Design
- ✅ Class-based (cleaner API, stateful)
- ✅ Type-safe config (dataclass)
- ✅ Separation of concerns

### More Control
- ✅ Configurable corruption type distribution
- ✅ Explicit parameters (not ranges)
- ✅ Easier to extend

### Compatibility
- ✅ Backward compatible (deprecated path)
- ✅ Stress tests updated
- ✅ No breaking changes to external APIs

---

## Future Work

Potential extensions:

1. **Dynamic corruption** - Adjust parameters during training
2. **Adaptive mixing** - Learn corruption type distribution
3. **Modality-specific parameters** - Different scales per modality
4. **Curriculum learning** - Gradually increase corruption strength

---

## Questions?

See:
- `CORRUPTION_SEMANTICS.md` for detailed semantics
- `README.md` for usage examples
- `corruptions.py` for implementation

