import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from models.partial import PartialGeM, PartialMean2d
from models.encoder import CNXv2Block
from models.utils import ChannelLayerNorm2d


# -----------------------------------------------------------------------------
#                         Branch Separable Channel miXer
# -----------------------------------------------------------------------------
class BSC_X(nn.Module):
    """
    Branch Separable miXer:
    Returns:
      logits (B, out_dim)
    """
    def __init__(
        self, 
        Cin_list: List[int],
        C=128, 
        use_block=True, 
        k_block=(5,1),
        p_block=(2,0),
        p_drop=0.2, 
        out_dim=128,
    ):
        super().__init__()
        self.Cin_list = list(Cin_list)
        self.proj = nn.Conv2d(sum(Cin_list), C, kernel_size=1, bias=False, groups=1)
        self.norm = ChannelLayerNorm2d(C, eps=1e-5)
        self.act  = nn.GELU()
        self.use_block = use_block
        self.block = CNXv2Block(C, k=k_block, p=p_block, s=1, expansion=2, p_drop=0.05) if use_block else nn.Identity()

        self.mean = PartialMean2d()
        self.gem = PartialGeM()
        self.vec_ln = nn.LayerNorm(2*C, eps=1e-5)
        self.vec_fc = nn.Sequential(
            nn.Linear(2*C, out_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p_drop) if p_drop > 0 else nn.Identity(),
        )
        self._last_aux = None

    @torch.no_grad()
    def _check_shapes(self, zs: List[torch.Tensor]):
        T0, F0 = zs[0].shape[-2], zs[0].shape[-1]
        for i, z in enumerate(zs):
            assert z.ndim == 4, f"zs[{i}] must be (B,C,T,F)"
            assert z.shape[-2] == T0 and z.shape[-1] == F0, "All streams must be aligned in (T,F)"

    def get_aux(self):
        return self._last_aux

    def forward(
            self,
            zs: List[torch.Tensor],
            ms: List[torch.Tensor],
    ) -> torch.Tensor:

        self._check_shapes(zs)
        Z = torch.cat(zs, dim=1)             # (B, sumCi, T, F)

        m1_list = []
        for z_i, m_i in zip(zs, ms):
            if m_i.shape[1] != 1:
                m_i = (m_i.sum(dim=1, keepdim=True) > 0).to(z_i.dtype)
            m1_list.append(m_i)
        M1 = torch.stack(m1_list, dim=0).amax(dim=0).to(Z.dtype)  # (B,1,T,F)

        Z = self.proj(Z)
        Z = self.norm(Z)
        Z = self.act(Z)

        M = M1.expand(-1, Z.shape[1], -1, -1)  # (B, C, T, F)
        Z = Z * M

        if self.use_block:
            Z, M_out = self.block(Z, M1) 
            M_exp = M_out.expand(-1, Z.shape[1], -1, -1)
        else:
            M_exp = M

        mean = self.mean(Z, M_exp)                        # (B, C)
        gem  = self.gem(Z, M_exp).squeeze(-1).squeeze(-1) # (B, C)
        vec  = torch.cat([mean, gem], dim=-1)             # (B, 2C)
        vec  = self.vec_ln(vec)
        vec  = self.vec_fc(vec)                           # (B, out_dim)

        with torch.no_grad():
            # overall mask coverage after OR over streams
            cov = M1.float().mean()                     # scalar
            # pooled feature stats
            norm_mean = mean.norm(dim=1).mean()         # scalar
            norm_gem  = gem.norm(dim=1).mean()          # scalar
            norm_vec  = vec.norm(dim=1).mean()          # scalar
            self._last_aux = {
                "mask_cov_overall": cov.detach().cpu(),
                "norm_mean": norm_mean.detach().cpu(),
                "norm_gem":  norm_gem.detach().cpu(),
                "norm_vec":  norm_vec.detach().cpu(),
            }

        return vec


# -----------------------------------------------------------------------------
#                            Baseline: ConcatProjFusion
# -----------------------------------------------------------------------------
class ConcatProjFusion(nn.Module):
    """
    Baseline: mean pool each stream independently, concatenate, and project the results.
    No cross-stream projection before pooling.
    """
    def __init__(self, Cin_list: List[int], out_dim=128):
        super().__init__()
        self.Cin_list = list(Cin_list)
        self.mean = PartialMean2d()

        self.vec_in = sum(Cin_list)
        self.ln = nn.LayerNorm(self.vec_in, eps=1e-5)
        self.fc = nn.Sequential(
            nn.Linear(self.vec_in, out_dim, bias=True),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, zs: List[torch.Tensor], ms: List[torch.Tensor]) -> torch.Tensor:
        vec = []
        for z_i, m_i in zip(zs, ms):
            if m_i.shape[1] != 1:
                m1 = (m_i.sum(dim=1, keepdim=True) > 0).to(z_i.dtype)
            else:
                m1 = m_i
            M = m1.expand(-1, z_i.shape[1], -1, -1)
            feat = self.mean(z_i, M)
            vec.append(feat)
        vec = torch.cat(vec, dim=1)
        return self.fc(self.ln(vec))


# -----------------------------------------------------------------------------
#                         Baseline: UniformAvgFusion
# -----------------------------------------------------------------------------
class UniformAvgFusion(nn.Module):
    """
    Baseline: pool each stream, project to unified vector space, then average pooled vectors.
    """
    def __init__(self, Cin_list: List[int], out_dim: int = 128):
        super().__init__()
        self.mean = PartialMean2d()
        self.proj = nn.ModuleList([
            (nn.Identity() if Cin == out_dim else nn.Linear(Cin, out_dim, bias=False))
            for Cin in Cin_list
        ])
        for lin in self.proj:
            self._orthogonal_rows_(lin)

    @staticmethod
    def _orthogonal_rows_(lin: nn.Linear):
        if not isinstance(lin, nn.Linear): return
        with torch.no_grad():
            # out x in
            w = torch.empty_like(lin.weight)
            nn.init.orthogonal_(w, gain=1.0)  # preserves scale
            lin.weight.copy_(w)

    def forward(self, zs: List[torch.Tensor], ms: List[torch.Tensor]) -> torch.Tensor:
        vecs = []
        avail = []
        for z_i, m_i, proj_i in zip(zs, ms, self.proj):
            if m_i.shape[1] != 1:
                m1 = (m_i.sum(dim=1, keepdim=True) > 0).to(z_i.dtype)
            else:
                m1 = m_i
            M = m1.expand(-1, z_i.shape[1], -1, -1)
            feat = self.mean(z_i, M)
            feat = proj_i(feat)
            present = (m1.sum(dim=(2,3)) > 0).to(feat.dtype)
            feat = F.normalize(feat, dim=-1, eps=1e-6)
            feat = feat * present
            vecs.append(feat)
            avail.append(present)
        A = torch.cat(avail, dim=1)
        denom = A.sum(dim=1, keepdim=True).clamp_min(1.0)
        w = A / denom
        vec = torch.stack(vecs, dim=1)
        vec = (vec * w.unsqueeze(-1)).sum(dim=1)
        return vec


# -----------------------------------------------------------------------------
#                             Baseline: BSXProjOnly
# -----------------------------------------------------------------------------
class BSXProjOnly(nn.Module):
    """
    BSX with only 1x1 projection + mask-aware pooling (no CNXv2 block).
    """
    def __init__(self, Cin_list: List[int], C: int = 128, out_dim: int = 128):
        super().__init__()
        self.core = BSC_X(
            Cin_list=Cin_list, C=C, out_dim=out_dim,
            use_block=False,
        )

    def forward(self, zs: List[torch.Tensor], ms: List[torch.Tensor]) -> torch.Tensor:
        return self.core(zs, ms)

