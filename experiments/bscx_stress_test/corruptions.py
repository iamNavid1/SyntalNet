"""
Corruption functions for stress-testing multi-channel fusion modules.

All functions operate on lists of feature tensors (zs) and mask tensors (ms),
where each tensor has shape (B, C, T, F_) for batch, channels, time, features.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F


def _shift_time(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shift tensor along time dimension (dim=-2).
    
    Args:
        x: (B, C, T, F_) tensor
        shift: integer shift amount (positive = shift right, negative = shift left)
    
    Returns:
        Shifted tensor of same shape
    """
    if shift == 0:
        return x
    B, C, T, F_ = x.shape
    if shift > 0:
        # Shift right: pad left, crop right
        pad = (0, 0, shift, 0)  # (pad_left, pad_right, pad_top, pad_bottom) for (F_, T)
        x_pad = F.pad(x, pad)
        return x_pad[..., :T, :]
    else:
        # Shift left: pad right, crop left
        shift = -shift
        pad = (0, 0, 0, shift)
        x_pad = F.pad(x, pad)
        return x_pad[..., shift:, :]


@torch.no_grad()
def stream_dropout(
    zs: List[torch.Tensor], 
    ms: List[torch.Tensor], 
    p_drop: float
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Drop entire streams with probability p_drop per sample.
    
    For each sample in the batch, each stream is independently dropped with
    probability p_drop. Dropped streams have their features and masks zeroed.
    
    Args:
        zs: List of feature tensors, each (B, C, T, F_)
        ms: List of mask tensors, each (B, 1 or C, T, F_)
        p_drop: Dropout probability per stream per sample
    
    Returns:
        Modified zs and ms lists
    """
    if p_drop <= 0:
        return zs, ms
    
    B = zs[0].shape[0]
    num_streams = len(zs)
    device = zs[0].device
    
    # Sample keep mask: (B, num_streams)
    keep = torch.rand((B, num_streams), device=device) > p_drop
    
    zs_out, ms_out = [], []
    for s, (z, m) in enumerate(zip(zs, ms)):
        # Expand keep mask to match tensor shape: (B, 1, 1, 1)
        k = keep[:, s].view(B, 1, 1, 1).to(z.dtype)
        zs_out.append(z * k)
        
        # Handle mask: collapse to single channel if needed
        if m.shape[1] != 1:
            m1 = (m.sum(dim=1, keepdim=True) > 0).to(m.dtype)
        else:
            m1 = m
        ms_out.append(m1 * k.to(m1.dtype))
    
    return zs_out, ms_out


@torch.no_grad()
def channel_dropout(
    zs: List[torch.Tensor], 
    ms: List[torch.Tensor], 
    drop_frac: float
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Drop a fraction of channels per stream uniformly at random.
    
    For each stream, randomly selects drop_frac of channels and zeros them.
    The same channels are dropped across all samples in the batch.
    
    Args:
        zs: List of feature tensors, each (B, C, T, F_)
        ms: List of mask tensors (unchanged)
        drop_frac: Fraction of channels to drop (0.0 to 1.0)
    
    Returns:
        Modified zs list, unchanged ms list
    """
    if drop_frac <= 0:
        return zs, ms
    
    zs_out = []
    for z in zs:
        B, C, T, F_ = z.shape
        n_drop = max(1, int(C * drop_frac))
        device = z.device
        
        # Randomly select channels to drop
        idx = torch.randperm(C, device=device)[:n_drop]
        
        # Create channel mask
        mask_c = torch.ones(C, device=device, dtype=z.dtype)
        mask_c[idx] = 0.0
        
        # Apply mask: (1, C, 1, 1) broadcasts to (B, C, T, F_)
        zs_out.append(z * mask_c.view(1, C, 1, 1))
    
    return zs_out, ms


@torch.no_grad()
def temporal_band_mask(
    zs: List[torch.Tensor], 
    ms: List[torch.Tensor], 
    band_frac: float = 0.2
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Zero a contiguous temporal band (same band for all streams per sample).
    
    For each sample, selects a random contiguous temporal band covering
    band_frac of the time dimension and zeros it in both features and masks.
    The same band is used across all streams for a given sample.
    
    Args:
        zs: List of feature tensors, each (B, C, T, F_)
        ms: List of mask tensors, each (B, 1 or C, T, F_)
        band_frac: Fraction of temporal dimension to mask (0.0 to 1.0)
    
    Returns:
        Modified zs and ms lists
    """
    if band_frac <= 0:
        return zs, ms
    
    B = zs[0].shape[0]
    T = zs[0].shape[-2]
    device = zs[0].device
    
    # Compute band length
    L = max(1, int(T * band_frac))
    
    # Sample start positions per batch: (B,)
    start = torch.randint(0, max(1, T - L + 1), (B,), device=device)
    
    zs_out, ms_out = [], []
    for z, m in zip(zs, ms):
        z2 = z.clone()
        m2 = m.clone()
        
        # Apply mask per sample
        for b in range(B):
            a = start[b].item()
            z2[b, :, a:a+L, :] = 0
            m2[b, :, a:a+L, :] = 0
        
        zs_out.append(z2)
        ms_out.append(m2)
    
    return zs_out, ms_out


@torch.no_grad()
def misalignment_jitter(
    zs: List[torch.Tensor], 
    ms: List[torch.Tensor], 
    max_shift: int = 2
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Shift each stream along time by a random integer in [-max_shift, max_shift].
    
    Each stream is independently shifted by a random amount. This simulates
    temporal misalignment between modalities.
    
    Args:
        zs: List of feature tensors, each (B, C, T, F_)
        ms: List of mask tensors, each (B, 1 or C, T, F_)
        max_shift: Maximum absolute shift amount (non-negative integer)
    
    Returns:
        Modified zs and ms lists
    """
    if max_shift <= 0:
        return zs, ms
    
    zs_out, ms_out = [], []
    for z, m in zip(zs, ms):
        device = z.device
        # Sample shift per stream (same for all samples in batch)
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=device).item())
        
        zs_out.append(_shift_time(z, shift))
        ms_out.append(_shift_time(m, shift))
    
    return zs_out, ms_out


@torch.no_grad()
def energy_imbalance(
    zs: List[torch.Tensor], 
    ms: List[torch.Tensor],
    scale: float = 3.0, 
    stream_idx: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Multiply one stream by a scale factor to test normalization robustness.
    
    Scales all features in one stream by the given factor. This tests whether
    layer normalization and other normalization mechanisms can handle
    significant energy imbalances between streams.
    
    Args:
        zs: List of feature tensors, each (B, C, T, F_)
        ms: List of mask tensors (unchanged)
        scale: Multiplicative scale factor
        stream_idx: Index of stream to scale (None = random selection)
    
    Returns:
        Modified zs list, unchanged ms list
    """
    if len(zs) == 0:
        return zs, ms
    
    if stream_idx is None:
        device = zs[0].device
        stream_idx = int(torch.randint(0, len(zs), (1,), device=device).item())
    
    zs_out = [z.clone() for z in zs]
    zs_out[stream_idx] = zs_out[stream_idx] * scale
    
    return zs_out, ms


@torch.no_grad()
def feature_noise(
    zs: List[torch.Tensor],
    ms: List[torch.Tensor],
    noise_level: float = 0.2,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Add masked, variance-scaled Gaussian noise to each stream.
    
    This corruption adds Gaussian noise to feature maps, where the noise
    standard deviation is a fraction of the feature standard deviation.
    The noise respects masks, so padded regions are not corrupted.
    
    Args:
        zs: List of feature tensors, each (B, C, T, F_)
        ms: List of mask tensors, each (B, 1 or C, T, F_)
        noise_level: Relative noise std; 0.0 = no noise,
                    0.2 means sigma_noise ≈ 0.2 * feature_std (per batch).
    
    Returns:
        (zs_noisy, ms)  # masks unchanged
    """
    if noise_level <= 0.0:
        return zs, ms
    
    zs_noisy: List[torch.Tensor] = []
    
    for z, m in zip(zs, ms):
        B, C, T, F_ = z.shape
        z_dtype = z.dtype
        device = z.device
        
        # Build a single-channel mask M1: (B, 1, T, F_)
        if m.shape[1] != 1:
            M1 = (m.sum(dim=1, keepdim=True) > 0).to(z_dtype)  # OR across channels
        else:
            M1 = m.to(z_dtype)
        
        # Expand mask per-channel for applying noise
        M = M1.expand(-1, C, -1, -1)  # (B, C, T, F_)
        
        # Avoid using padded positions when computing stats
        valid = (M > 0.5)
        z_valid = z * valid
        
        # Per-batch mean over valid positions
        # Denominator: count of valid entries per batch
        denom = valid.view(B, -1).sum(dim=1).clamp_min(1.0)  # (B,)
        mean_b = z_valid.view(B, -1).sum(dim=1) / denom       # (B,)
        
        # Per-batch variance over valid positions
        z_centered = z_valid.view(B, -1) - mean_b.unsqueeze(1)  # (B, N)
        var_b = (z_centered.pow(2) * valid.view(B, -1)).sum(dim=1) / denom
        std_b = var_b.clamp_min(1e-6).sqrt()                   # (B,)
        
        # Noise std per batch
        sigma_b = noise_level * std_b                          # (B,)
        sigma = sigma_b.view(B, 1, 1, 1)
        
        # Sample noise and apply only where valid
        eps = torch.randn_like(z)
        noise = eps * sigma
        z_noisy = z + noise * M
        
        zs_noisy.append(z_noisy)
    
    return zs_noisy, ms

