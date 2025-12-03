from __future__ import annotations

from typing import List, Optional, Dict, Callable

import torch


Tensor = torch.Tensor


@torch.no_grad()
def dropout_corruption(
    z_list: List[Tensor],
    p_drop: float,
    which_mod: Optional[int] = None,
) -> List[Tensor]:
    """
    Per-modality dropout on pooled branch embeddings.

    For each selected modality j and sample b:
      - with prob p_drop: z[b,j] is zeroed out
      - with prob 1-p_drop: left unchanged
    """
    if p_drop <= 0.0:
        return z_list

    out: List[Tensor] = []
    M = len(z_list)
    for j, z in enumerate(z_list):
        if (which_mod is not None) and (j != which_mod):
            out.append(z)
            continue
        B = z.shape[0]
        device = z.device
        keep = (torch.rand(B, device=device) > p_drop).to(z.dtype).view(B, 1)
        out.append(z * keep)
    return out


@torch.no_grad()
def noise_corruption(
    z_list: List[Tensor],
    noise_level: float,
    which_mod: Optional[int] = None,
) -> List[Tensor]:
    """
    Add Gaussian noise to selected modalities, scaled to per-sample std.
    """
    if noise_level <= 0.0:
        return z_list

    out: List[Tensor] = []
    for j, z in enumerate(z_list):
        if (which_mod is not None) and (j != which_mod):
            out.append(z)
            continue
        B, D = z.shape
        device = z.device
        dtype = z.dtype
        mean = z.mean(dim=1, keepdim=True)
        var = ((z - mean) ** 2).mean(dim=1, keepdim=True)
        std = var.clamp_min(1e-6).sqrt()
        sigma = noise_level * std
        eps = torch.randn_like(z, device=device, dtype=dtype)
        out.append(z + eps * sigma)
    return out


@torch.no_grad()
def shuffle_corruption(
    z_list: List[Tensor],
    p_shuffle: float,
    which_mod: Optional[int] = None,
) -> List[Tensor]:
    """
    Shuffle a fraction of samples for the selected modality across the batch.
    """
    if p_shuffle <= 0.0:
        return z_list

    out: List[Tensor] = []
    for j, z in enumerate(z_list):
        if (which_mod is not None) and (j != which_mod):
            out.append(z)
            continue
        B = z.shape[0]
        device = z.device
        idx_corrupt = (torch.rand(B, device=device) < p_shuffle)
        perm = torch.randperm(B, device=device)
        z_perm = z[perm]
        z_new = z.clone()
        z_new[idx_corrupt] = z_perm[idx_corrupt]
        out.append(z_new)
    return out


@torch.no_grad()
def rescale_corruption(
    z_list: List[Tensor],
    scale: float,
    which_mod: Optional[int] = None,
) -> List[Tensor]:
    """
    Rescale the energy of selected modalities by a constant factor.
    """
    if scale == 1.0:
        return z_list
    out: List[Tensor] = []
    for j, z in enumerate(z_list):
        if (which_mod is not None) and (j != which_mod):
            out.append(z)
        else:
            out.append(z * scale)
    return out


CORRUPTION_FUNCS: Dict[str, Callable[..., List[Tensor]]] = {
    "dropout": dropout_corruption,
    "noise": noise_corruption,
    "shuffle": shuffle_corruption,
    "rescale": rescale_corruption,
}


