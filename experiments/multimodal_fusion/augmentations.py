from __future__ import annotations

from typing import List

import torch

from .config import ReliabilityAugmentationConfig


class ReliabilitySwitchAugmentor:
    """
    Applies the reliability-switch corruption strategy on branch embeddings.

    Exactly one modality per sample is corrupted whenever the Bernoulli draw
    (with probability `p_corrupt`) succeeds. The corruption type is drawn from
    the configured mixture (dropout / noise / shuffle).
    """

    def __init__(self, cfg: ReliabilityAugmentationConfig):
        self.cfg = cfg
        mix = torch.tensor(
            [
                max(float(cfg.mix.get("dropout", 0.0)), 0.0),
                max(float(cfg.mix.get("noise", 0.0)), 0.0),
                max(float(cfg.mix.get("shuffle", 0.0)), 0.0),
            ],
            dtype=torch.float32,
        )
        if mix.sum() == 0:
            mix[:] = 1.0
        self.mix = mix / mix.sum()
        self.enabled = True

    def enable(self, flag: bool) -> None:
        self.enabled = flag

    def __call__(self, z_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if (not self.enabled) or self.cfg.p_corrupt <= 0.0:
            return z_list
        if not z_list:
            return z_list

        B = z_list[0].shape[0]
        device = z_list[0].device
        dtype = z_list[0].dtype
        corrupt_mask = torch.rand(B, device=device) < self.cfg.p_corrupt
        if not torch.any(corrupt_mask):
            return z_list

        mod_choices = torch.randint(
            low=0, high=len(z_list), size=(B,), device=device
        )
        type_choices = torch.multinomial(
            self.mix.to(device), num_samples=B, replacement=True
        )

        augmented = [z.clone() for z in z_list]

        for mod_idx, z in enumerate(augmented):
            idx_mod = corrupt_mask & (mod_choices == mod_idx)
            if not torch.any(idx_mod):
                continue

            self._apply_dropout(z, idx_mod, type_choices == 0, dtype)
            self._apply_noise(z, idx_mod, type_choices == 1, dtype)
            self._apply_shuffle(z, idx_mod, type_choices == 2, dtype)

        return augmented

    def _apply_dropout(
        self,
        z: torch.Tensor,
        mod_mask: torch.Tensor,
        type_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        mask = mod_mask & type_mask
        if not torch.any(mask):
            return
        strengths = self._sample_range(
            self.cfg.dropout_range, mask.sum().item(), z.device, dtype
        )
        z[mask] = z[mask] * (1.0 - strengths.unsqueeze(1))

    def _apply_noise(
        self,
        z: torch.Tensor,
        mod_mask: torch.Tensor,
        type_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        mask = mod_mask & type_mask
        if not torch.any(mask):
            return
        strengths = self._sample_range(
            self.cfg.noise_range, mask.sum().item(), z.device, dtype
        )
        sub = z[mask]
        if sub.numel() == 0:
            return
        std = sub.std(dim=1, keepdim=True).clamp_min(1e-6)
        eps = torch.randn_like(sub)
        z[mask] = sub + eps * std * strengths.unsqueeze(1)

    def _apply_shuffle(
        self,
        z: torch.Tensor,
        mod_mask: torch.Tensor,
        type_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        mask = mod_mask & type_mask
        if not torch.any(mask):
            return
        strengths = self._sample_range(
            self.cfg.shuffle_range, mask.sum().item(), z.device, dtype
        )
        B = z.shape[0]
        perm = torch.randperm(B, device=z.device)
        shuffled = z[perm][mask]
        src = z[mask]
        mix = strengths.unsqueeze(1)
        z[mask] = (1.0 - mix) * src + mix * shuffled

    @staticmethod
    def _sample_range(
        bounds, count: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        low, high = bounds if isinstance(bounds, (list, tuple)) else (bounds, bounds)
        if low == high:
            return torch.full((count,), float(low), device=device, dtype=dtype)
        return torch.empty(count, device=device, dtype=dtype).uniform_(float(low), float(high))

