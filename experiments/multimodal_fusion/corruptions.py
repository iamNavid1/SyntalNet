from __future__ import annotations
from typing import List, Tuple, Optional
import torch


@torch.no_grad()
def modality_dropout(
    z_list: List[torch.Tensor],
    avail: Optional[torch.Tensor],
    p_drop: float,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Drop entire modalities with probability p_drop per sample.
    
    For each sample and modality, with probability p_drop:
    - Zero the modality vector z_i
    - Set availability to 0
    
    Args:
        z_list: List of (B, D) modality feature tensors
        avail: (B, M) availability mask or None
        p_drop: Dropout probability per modality per sample
    
    Returns:
        (z_list_dropped, avail_dropped)
    """
    if p_drop <= 0.0:
        return z_list, avail
    
    M = len(z_list)
    B = z_list[0].shape[0]
    device = z_list[0].device
    dtype = z_list[0].dtype
    
    if avail is None:
        avail = torch.ones(B, M, device=device, dtype=dtype)
    
    # Sample keep mask: (B, M)
    keep = (torch.rand(B, M, device=device) > p_drop).to(dtype)
    
    # If all modalities are dropped, randomly keep one
    all_dropped = (keep.sum(dim=1) == 0)  # (B,)
    if all_dropped.any():
        for b in range(B):
            if all_dropped[b]:
                mod_to_keep = int(torch.randint(0, M, (1,), device=device).item())
                keep[b, mod_to_keep] = 1.0
    
    avail_new = avail * keep
    
    z_out: List[torch.Tensor] = []
    for j, z in enumerate(z_list):
        k = keep[:, j].view(B, 1)  # (B, 1)
        z_out.append(z * k)
    
    return z_out, avail_new


@torch.no_grad()
def modality_noise(
    z_list: List[torch.Tensor],
    avail: Optional[torch.Tensor],
    noise_level: float = 0.2,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Add Gaussian noise to each modality vector, scaled to feature std.
    
    For each modality:
    - Compute per-sample feature std
    - Add noise: z_i' = z_i + noise_level * std(z_i) * ε
    - Availability remains unchanged (modality is present but degraded)
    
    Args:
        z_list: List of (B, D) modality feature tensors
        avail: (B, M) availability mask or None (unchanged)
        noise_level: Relative noise std; 0.2 means sigma ≈ 0.2 * std(z)
    
    Returns:
        (z_noisy_list, avail)
    """
    if noise_level <= 0.0:
        return z_list, avail
    
    z_noisy: List[torch.Tensor] = []
    
    for z in z_list:
        B, D = z.shape
        device = z.device
        dtype = z.dtype
        
        # Per-sample std
        mean = z.mean(dim=1, keepdim=True)  # (B, 1)
        var = ((z - mean) ** 2).mean(dim=1, keepdim=True)  # (B, 1)
        std = var.clamp_min(1e-6).sqrt()  # (B, 1)
        
        sigma = noise_level * std  # (B, 1)
        eps = torch.randn_like(z)
        noise = eps * sigma
        z_noisy.append(z + noise)
    
    return z_noisy, avail


@torch.no_grad()
def modality_shuffle(
    z_list: List[torch.Tensor],
    avail: Optional[torch.Tensor],
    p_shuffle: float = 0.5,
    which_mod: Optional[int] = None,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Shuffle one modality across the batch (contradicting other modalities).
    
    This creates cross-modal conflicts:
    - One modality is shuffled across samples
    - Labels and other modalities remain aligned
    - Tests robustness to contradictory modality information
    
    Args:
        z_list: List of (B, D) modality feature tensors
        avail: (B, M) availability mask or None (unchanged)
        p_shuffle: Fraction of samples to corrupt
        which_mod: Index of modality to shuffle; None = random selection
    
    Returns:
        (z_shuffled_list, avail)
    """
    if p_shuffle <= 0.0:
        return z_list, avail
    
    M = len(z_list)
    B = z_list[0].shape[0]
    device = z_list[0].device
    
    if which_mod is None:
        which_mod = int(torch.randint(low=0, high=M, size=(1,), device=device).item())
    
    z_out: List[torch.Tensor] = []
    for j, z in enumerate(z_list):
        if j != which_mod:
            z_out.append(z)
            continue
        
        # Create index mask of corrupted samples
        idx_corrupt = (torch.rand(B, device=device) < p_shuffle)
        perm = torch.randperm(B, device=device)
        z_perm = z[perm]
        
        z_new = z.clone()
        z_new[idx_corrupt] = z_perm[idx_corrupt]
        z_out.append(z_new)
    
    return z_out, avail


@torch.no_grad()
def modality_rescale(
    z_list: List[torch.Tensor],
    avail: Optional[torch.Tensor],
    scale: float = 4.0,
    which_mod: Optional[int] = None,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Multiply one modality vector by a scale factor (energy imbalance).
    
    Tests robustness to:
    - Varying feature magnitudes across modalities
    - Normalization effectiveness
    - Allocation mechanism's ability to handle imbalanced energies
    
    Args:
        z_list: List of (B, D) modality feature tensors
        avail: (B, M) availability mask or None (unchanged)
        scale: Multiplicative scale factor
        which_mod: Index of modality to scale; None = random selection
    
    Returns:
        (z_scaled_list, avail)
    """
    if len(z_list) == 0 or scale == 1.0:
        return z_list, avail
    
    M = len(z_list)
    B = z_list[0].shape[0]
    device = z_list[0].device
    
    if which_mod is None:
        which_mod = int(torch.randint(low=0, high=M, size=(1,), device=device).item())
    
    z_out = [z.clone() for z in z_list]
    z_out[which_mod] = z_out[which_mod] * scale
    return z_out, avail


# Mapping from corruption name to function
CORRUPTION_FUNCS = {
    "modality_dropout": modality_dropout,
    "modality_noise": modality_noise,
    "modality_shuffle": modality_shuffle,
    "modality_rescale": modality_rescale,
}


# Baseline parameter values (no corruption)
BASELINE_PARAMS = {
    "modality_dropout": 0.0,
    "modality_noise": 0.0,
    "modality_shuffle": 0.0,
    "modality_rescale": 1.0,
}

