"""
Corruption utilities for reliability-switch training and stress testing.

Provides a class-based ReliabilitySwitchCorruptor for training-time augmentation
and functional helpers for stress testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


# =============================================================================
#                           Configuration
# =============================================================================

@dataclass
class ReliabilitySwitchConfig:
    """Configuration for reliability-switch corruption during training."""
    
    p_corrupt: float = 0.7  # Probability of corrupting a sample
    
    # Corruption type mixture (should sum to 1.0, will be normalized)
    mix_dropout: float = 0.4
    mix_noise: float = 0.4
    mix_shuffle: float = 0.2
    
    # Corruption parameters
    dropout_scale: float = 1.0  # Scale factor in [0, 1] for modality-level dropout
    noise_level: float = 0.35   # Multiplier for per-sample std
    shuffle_fraction: float = 0.5  # Probability for Bernoulli mask in shuffle
    
    @classmethod
    def from_dict(cls, cfg: dict) -> "ReliabilitySwitchConfig":
        """Create config from dictionary (for YAML loading)."""
        corruption_strengths = cfg.get("corruption_strengths", {})
        
        # Extract mix if provided
        mix = cfg.get("corruption_mix", {})
        
        return cls(
            p_corrupt=float(cfg.get("p_corrupt", 0.7)),
            mix_dropout=float(mix.get("dropout", 0.4)) if mix else 0.4,
            mix_noise=float(mix.get("noise", 0.4)) if mix else 0.4,
            mix_shuffle=float(mix.get("shuffle", 0.2)) if mix else 0.2,
            dropout_scale=float(cfg.get("dropout_scale", 1.0)),
            noise_level=float(cfg.get("noise_level", 0.35)),
            shuffle_fraction=float(cfg.get("shuffle_fraction", 0.5)),
        )


# =============================================================================
#                    Reliability-Switch Corruptor (Class-based)
# =============================================================================

class ReliabilitySwitchCorruptor:
    """
    Applies reliability-switch augmentation during training:
      - For each sample, with probability p_corrupt pick exactly one modality
      - Sample a corruption type (dropout / noise / shuffle) from given mix
      - Apply corruption to the selected modality embedding
    
    Corruption semantics:
      - Dropout: Modality-level scaling (multiply whole embedding by 1 - dropout_scale)
      - Noise: Per-sample statistics (add Gaussian noise scaled by sample std)
      - Shuffle: Cross-sample contradictions (mix with different sample via Bernoulli mask)
    """
    
    def __init__(self, cfg: ReliabilitySwitchConfig):
        self.cfg = cfg
        
        # Normalize corruption type mixture
        mix = torch.tensor(
            [cfg.mix_dropout, cfg.mix_noise, cfg.mix_shuffle],
            dtype=torch.float32,
        )
        mix = torch.clamp(mix, min=0.0)
        if mix.sum() == 0:
            mix = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        self.mix = mix / mix.sum()
    
    def __call__(self, z_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply reliability-switch corruption to branch embeddings.
        
        Args:
            z_list: List of M branch embeddings, each of shape (B, D)
        
        Returns:
            List of corrupted embeddings
        """
        if not z_list or len(z_list) == 1:
            return z_list
        
        B = z_list[0].shape[0]
        device = z_list[0].device
        dtype = z_list[0].dtype
        
        if self.cfg.p_corrupt <= 0:
            return z_list
        
        # Determine which samples get corrupted
        corrupt_mask = torch.rand(B, device=device) < self.cfg.p_corrupt
        idx = torch.nonzero(corrupt_mask, as_tuple=False).squeeze(-1)
        
        if idx.numel() == 0:
            return z_list
        
        M = len(z_list)
        z_out = [z.clone() for z in z_list]
        
        # Sample modality + corruption type per selected sample
        mod_choices = torch.randint(0, M, (idx.numel(),), device=device)
        type_choices = torch.multinomial(
            self.mix.to(device), idx.numel(), replacement=True
        )
        
        # Create selection tensors for the full batch
        selected_mod = torch.full((B,), -1, dtype=torch.long, device=device)
        selected_type = torch.full((B,), -1, dtype=torch.long, device=device)
        selected_mod[idx] = mod_choices
        selected_type[idx] = type_choices
        
        # Apply corruptions per modality
        for mod_idx in range(M):
            mask_mod = selected_mod == mod_idx
            if not mask_mod.any():
                continue
            
            source = z_list[mod_idx]
            target = z_out[mod_idx]
            
            # Dropout (type 0): Modality-level scaling
            drop_idx = torch.nonzero(
                mask_mod & (selected_type == 0), as_tuple=False
            ).squeeze(-1)
            if drop_idx.numel() > 0:
                scale = float(
                    torch.clamp(torch.tensor(self.cfg.dropout_scale), 0.0, 1.0)
                )
                target[drop_idx] = target[drop_idx] * (1.0 - scale)
            
            # Noise (type 1): Per-sample statistics
            noise_idx = torch.nonzero(
                mask_mod & (selected_type == 1), as_tuple=False
            ).squeeze(-1)
            if noise_idx.numel() > 0 and self.cfg.noise_level > 0:
                subset = target[noise_idx]
                # Compute per-sample mean and std
                mean = subset.mean(dim=1, keepdim=True)
                std = (
                    ((subset - mean) ** 2).mean(dim=1, keepdim=True)
                    .clamp_min(1e-6)
                    .sqrt()
                )
                eps = torch.randn_like(subset)
                target[noise_idx] = subset + self.cfg.noise_level * std * eps
            
            # Shuffle (type 2): Cross-sample contradictions
            shuffle_idx = torch.nonzero(
                mask_mod & (selected_type == 2), as_tuple=False
            ).squeeze(-1)
            if shuffle_idx.numel() > 0 and self.cfg.shuffle_fraction > 0:
                # Draw different samples from batch
                perm = torch.randint(0, B, (shuffle_idx.numel(),), device=device)
                
                # Avoid trivial permutation (same index)
                same = perm == shuffle_idx
                if same.any():
                    perm[same] = (
                        perm[same] + torch.randint(1, B, (same.sum(),), device=device)
                    ) % B
                
                perm_vals = source[perm]
                
                # Bernoulli mask for partial mixing
                frac = torch.clamp(
                    torch.tensor(self.cfg.shuffle_fraction, device=device, dtype=dtype),
                    0.0, 1.0
                )
                mask = (torch.rand_like(perm_vals) < frac).to(dtype)
                target[shuffle_idx] = (
                    mask * perm_vals + (1.0 - mask) * target[shuffle_idx]
                )
            
            z_out[mod_idx] = target
        
        return z_out


# =============================================================================
#                    Individual Corruption Functions
# =============================================================================

def corrupt_branch_dropout(z: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply modality-level dropout scaling.
    
    Args:
        z: Branch embedding of shape (B, D)
        p: Dropout scale in [0, 1] (multiply by 1-p)
        
    Returns:
        Corrupted embedding (scaled by 1-p)
    """
    if p <= 0.0:
        return z
    
    scale = 1.0 - torch.clamp(torch.tensor(p), 0.0, 1.0).item()
    return z * scale


def corrupt_branch_noise(z: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Add Gaussian noise using per-sample statistics.
    
    Args:
        z: Branch embedding of shape (B, D)
        scale: Noise level (multiplier for per-sample std)
        
    Returns:
        Corrupted embedding
    """
    if scale <= 0.0:
        return z
    
    # Compute per-sample mean and std
    mean = z.mean(dim=1, keepdim=True)
    std = ((z - mean) ** 2).mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()
    
    # Generate noise scaled by per-sample std
    eps = torch.randn_like(z)
    return z + scale * std * eps


def corrupt_branch_shuffle(z: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply cross-sample shuffle with Bernoulli mixing.
    
    Args:
        z: Branch embedding of shape (B, D)
        p: Shuffle fraction (probability for Bernoulli mask)
        
    Returns:
        Corrupted embedding
    """
    if p <= 0.0:
        return z
    
    B, D = z.shape
    device = z.device
    dtype = z.dtype
    
    # Permute all samples
    perm = torch.randperm(B, device=device)
    
    # Avoid trivial permutation where possible
    same = perm == torch.arange(B, device=device)
    if same.any() and B > 1:
        # Swap with next index (circular)
        perm[same] = (perm[same] + 1) % B
    
    z_perm = z[perm]
    
    # Bernoulli mask for partial mixing
    frac = torch.clamp(torch.tensor(p, device=device, dtype=dtype), 0.0, 1.0)
    mask = (torch.rand_like(z) < frac).to(dtype)
    
    return mask * z_perm + (1.0 - mask) * z


# =============================================================================
#                    Stress Test Corruption Functions
# =============================================================================

def apply_uniform_corruption(
    z_list: List[torch.Tensor],
    corruption_type: str,
    corruption_param: float,
) -> List[torch.Tensor]:
    """
    Apply a single corruption type uniformly to all modalities.
    Used for stress testing.
    
    Semantics match ReliabilitySwitchCorruptor:
      - Dropout: modality-level scaling
      - Noise: per-sample statistics
      - Shuffle: cross-sample Bernoulli mixing
    
    Args:
        z_list: List of branch embeddings
        corruption_type: One of "dropout", "noise", "shuffle"
        corruption_param: Corruption strength
        
    Returns:
        List of corrupted embeddings
    """
    corruption_type = corruption_type.lower().strip()
    
    if corruption_type == "dropout":
        return [corrupt_branch_dropout(z, corruption_param) for z in z_list]
    elif corruption_type == "noise":
        return [corrupt_branch_noise(z, corruption_param) for z in z_list]
    elif corruption_type == "shuffle":
        return [corrupt_branch_shuffle(z, corruption_param) for z in z_list]
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


def apply_single_modality_corruption(
    z_list: List[torch.Tensor],
    modality_idx: int,
    corruption_type: str,
    corruption_param: float,
) -> List[torch.Tensor]:
    """
    Apply corruption to a single modality (for per-modality stress tests).
    
    Semantics match ReliabilitySwitchCorruptor.
    
    Args:
        z_list: List of branch embeddings
        modality_idx: Index of modality to corrupt
        corruption_type: One of "dropout", "noise", "shuffle"
        corruption_param: Corruption strength
        
    Returns:
        List of embeddings with one modality corrupted
    """
    z_corrupted = [z.clone() for z in z_list]
    
    corruption_type = corruption_type.lower().strip()
    
    if corruption_type == "dropout":
        z_corrupted[modality_idx] = corrupt_branch_dropout(
            z_list[modality_idx], corruption_param
        )
    elif corruption_type == "noise":
        z_corrupted[modality_idx] = corrupt_branch_noise(
            z_list[modality_idx], corruption_param
        )
    elif corruption_type == "shuffle":
        z_corrupted[modality_idx] = corrupt_branch_shuffle(
            z_list[modality_idx], corruption_param
        )
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    return z_corrupted


# =============================================================================
#                    Legacy Compatibility (DEPRECATED)
# =============================================================================

def apply_reliability_switch_corruption(
    z_list: List[torch.Tensor],
    p_corrupt: float = 0.7,
    corruption_strengths: Optional[dict] = None,
) -> List[torch.Tensor]:
    """
    DEPRECATED: Use ReliabilitySwitchCorruptor class instead.
    
    Legacy function for backward compatibility. Creates a default config
    and uses the class-based corruptor.
    """
    if corruption_strengths is None:
        corruption_strengths = {
            'dropout': (0.1, 0.5),
            'noise': (0.1, 0.5),
            'shuffle': (0.1, 0.5),
        }
    
    # Create config with average of ranges
    cfg = ReliabilitySwitchConfig(
        p_corrupt=p_corrupt,
        dropout_scale=sum(corruption_strengths['dropout']) / 2.0,
        noise_level=sum(corruption_strengths['noise']) / 2.0,
        shuffle_fraction=sum(corruption_strengths['shuffle']) / 2.0,
    )
    
    corruptor = ReliabilitySwitchCorruptor(cfg)
    return corruptor(z_list)
