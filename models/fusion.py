import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import Tuple, Optional, Dict, Any, List

from partial import PartialConv2d, PartialAvgPool2d, PartialGeM
from utils import ChannelLayerNorm2d


class DepthwiseSeparable(nn.Module):
    """
    DW 3x3 + PW 1x1 
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = PartialConv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False, return_mask=False)
        self.pw = PartialConv2d(in_ch, out_ch, kernel_size=1, bias=False, return_mask=True)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m_grp = m.expand(-1, x.shape[1], -1, -1) if m.shape[1] == 1 else m
        x = self.dw(x, m_grp)
        x, m = self.pw(x, m)
        return x, m


class ResidualPreNorm(nn.Module):
    """
    Pre-norm residual: LN -> SiLU -> optional scaling -> body -> Dropout -> add skip.
    """
    def __init__(
            self,
            in_ch: int, 
            out_ch: int, 
            body: nn.Module, 
            p_drop: float = 0.1, 
            gated: bool = False,
            mask_aware_skip: bool = True,
            skip_stride: int | tuple[int,int] = 1,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.gated = gated
        self.mask_aware_skip = mask_aware_skip

        self.ln    = ChannelLayerNorm2d(in_ch)
        self.act   = nn.SiLU(inplace=True)
        self.body  = body
        self.drop  = nn.Dropout(p_drop)

        self.skip_stride = _pair(skip_stride)
        if not self.mask_aware_skip:  # vanilla skip
            self._skip_mode = "vanilla"
            self.skip = (
                nn.Identity() if (in_ch == out_ch and self.skip_stride == (1,1)) \
                else nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False)
            )
        else:  # mask-aware skip
            self._skip_mode = "partial_identity" if (in_ch == out_ch and self.skip_stride == (1,1)) else "partial_1x1"
            need_mask_update = (self.skip_stride != (1,1))
            self.skip = (
                None if (in_ch == out_ch and self.skip_stride == (1,1)) \
                else PartialConv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False, return_mask=need_mask_update)
            )

    @staticmethod
    def _to_shared_mask(m: torch.Tensor, dtype) -> torch.Tensor:
        return m if m.shape[1] == 1 else (m.sum(dim=1, keepdim=True) > 0).to(dtype)

    @staticmethod
    def _expand_to_channels(m: torch.Tensor, C: int, dtype) -> torch.Tensor:
        return m if m.shape[1] == C else m.expand(-1, C, -1, -1).to(dtype)

    def _skip_forward(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._skip_mode == "vanilla":
            return self.skip(x), m

        if self._skip_mode == "partial_identity":
            m_exp = self._expand_to_channels(m, x.shape[1], x.dtype)
            return x * m_exp, m

        # partial 1x1 skip
        out = self.skip(x, m)
        return out if isinstance(out, tuple) else (out, m)

    def forward(self, x: torch.Tensor, m: torch.Tensor, g1: Optional[torch.Tensor] = None, g2: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1) skip
        s, m_s = self._skip_forward(x, m)

        # 2) pre-norm + optional gating)
        y = self.act(self.ln(x))
        if self.gated:
            assert (g1 is not None) and (g2 is not None), "gate values must be provided for gated mode"
            assert self.in_ch % 2 == 0, "halves-gate expects even in_ch"
            y_1  = y[:, :self.in_ch // 2] * g1
            y_2 = y[:, self.in_ch // 2:] * g2
            y = torch.cat([y_1, y_2], dim=1)

        # 3) main body: returns (features, 1-ch mask)
        y, m_body = self.body(y, m)

        # 4) dropout + residual sum
        y = self.drop(y)
        out = s + y

        # 5) combine masks with OR
        m_s_shared = self._to_shared_mask(m_s, out.dtype)
        m_out = torch.maximum(m_s_shared, m_body)   # (B,1,H,W)

        return out, m_out


class ModalityGate(nn.Module):
    """
    Computes sample-wise scalars g_1, g_2 for two modalities:
      - w = softmax(head_alloc(s)) in R^2  (relative allocation)
      - q = sigmoid(head_qual(s)) in (0,1) (overall quality)
      - g = eps + (1 - eps) * q * w        (floor to avoid dead paths)
    """
    def __init__(
            self,
            C_mod: int,
            eps: float = 0.05,
            init_quality: float = 0.8,
            init_alloc_bias: float = 0.0,
    ):
        super().__init__()
        assert 0.0 < eps < 0.5, "eps should be a small floor (e.g., 0.02..0.1)"
        self.eps = eps
        self.pool = PartialGeM()
        self.head = nn.Linear(2 * C_mod, 3, bias=True)  # 2 for alloc, 1 for quality

        # Initialize biases
        with torch.no_grad():
            b = self.head.bias
            # [alloc_kp, alloc_vid, qual]
            b.zero_()
            if init_alloc_bias != 0.0:
                b[0] =  init_alloc_bias
                b[1] = -init_alloc_bias
            b[2] = math.log(init_quality / (1.0 - init_quality))

    def forward(
            self,
            z_1: torch.Tensor, z_2: torch.Tensor, 
            m_1: torch.Tensor, m_2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # s_*: (B, C_mod)
        s_1  = self.pool(z_1, m_1).flatten(1)
        s_2 = self.pool(z_2, m_2).flatten(1)
        s = torch.cat([s_1, s_2], dim=1)   # (B, 2*C_mod)

        logits = self.head(s)                 # (B, 3)
        alloc_logits = logits[:, :2]          # (B, 2)
        qual_logit   = logits[:, 2:3]         # (B, 1)

        w = F.softmax(alloc_logits, dim=-1)   # (B, 2)
        q = torch.sigmoid(qual_logit)         # (B, 1)

        g = self.eps + (1.0 - self.eps) * q * w   # (B, 2)
        g_1  = g[:, 0:1].view(-1, 1, 1, 1)         # (B,1,1,1)
        g_2  = g[:, 1:2].view(-1, 1, 1, 1)         # (B,1,1,1)
        return g_1, g_2, w, q


class GPSFusion(nn.Module):
    """
    Gated Partial Separable Fusion:
    Inputs:  z_kp, z_vid  each (B, C_mod, T, F)
    Pipeline:
      - LN per modality
      - Two-factor gate (quality * allocation) from normalized streams
      - Concat on C
      - ResidualPreNorm (apply gating)
      - AvgPool2d (1,2) on F
      - ResidualPreNorm (DW+PW)
      - GeM -> Dropout -> Linear head

    Returns:
      logits (B, out_dim) and (optionally) a dict with gate stats for logging.
    """
    def __init__(self,
                 C_mod: int,
                 C_exp: List[int] = [2, 2],
                 p_drop_res: float = 0.1,
                 p_drop_head: float = 0.2,
                 gate_eps: float = 0.05,
                 gate_init_quality: float = 0.8,
                 gate_init_alloc_bias: float = 0.0,
                 return_aux: bool = False,
    ):
        super().__init__()
        assert len(C_exp) == 2, f"C_exp must be of length 2, got {len(C_exp)}"

        self.C_mod = C_mod
        self.C_tot = 2 * C_mod
        self.return_aux = return_aux

        # pre-norm per modality
        self.ln_1 = ChannelLayerNorm2d(self.C_mod)
        self.ln_2 = ChannelLayerNorm2d(self.C_mod)

        # two-factor gate
        self.mod_gate = ModalityGate(
            C_mod=self.C_mod,
            eps=gate_eps,
            init_quality=gate_init_quality,
            init_alloc_bias=gate_init_alloc_bias,
        )

        # Block 1: gating inside (after LN, before mixing)
        self.block1 = ResidualPreNorm(
            in_ch=self.C_tot, out_ch=self.C_tot*C_exp[0],
            body=DepthwiseSeparable(self.C_tot, self.C_tot*C_exp[0]),
            p_drop=p_drop_res, gated=True,
        )

        # reduce feature width (F) by 2
        self.poolF = PartialAvgPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Block 2: standard pre-norm residual
        self.block2 = ResidualPreNorm(
            in_ch=self.C_tot*C_exp[0], out_ch=self.C_tot*C_exp[0]*C_exp[1],
            body=DepthwiseSeparable(self.C_tot*C_exp[0], self.C_tot*C_exp[0]*C_exp[1]),
            p_drop=p_drop_res, gated=False,
        )

        # GeM head
        self.gem  = PartialGeM()
        self.drop = nn.Dropout(p_drop_head)

    def forward(
            self,
            z_1: torch.Tensor, z_2: torch.Tensor, 
            m_1: torch.Tensor, m_2: torch.Tensor,
    ):
        # normalize each modality first for stable gate statistics
        z_1 = self.ln_1(z_1)    # (B, C_mod, T, F)
        z_2 = self.ln_2(z_2)     # (B, C_mod, T, F)

        m_1 = m_1.expand(-1, z_1.shape[1], -1, -1)
        m_2 = m_2.expand(-1, z_2.shape[1], -1, -1)

        # two-factor gate
        g_1, g_2, w, q = self.mod_gate(z_1, z_2, m_1, m_2)  # each scalar per sample

        # concat
        z = torch.cat([z_1, z_2], dim=1)           # (B, 2*C_mod, T, F)
        m = torch.cat([m_1, m_2], dim=1)           # (B, 2*C_mod, T, F)

        # Block 1: apply gating
        z, m = self.block1(z, m, g1=g_1, g2=g_2)   # z: (B, 4*C_mod, T, F);  m: (B, 1, T, F)

        # Block 2: pooling & light conv stack
        z, m = self.poolF(z, m)                    # z: (B, 4*C_mod, T, F/2);  m: (B, 1, T, F/2)
        z, m = self.block2(z, m)                   # z: (B, 8*C_mod, T, F/2);  m: (B, 1, T, F/2)

        # GeM head
        z = self.gem(z, m).flatten(1)              # (B, 8*C_mod)
        z = self.drop(z)

        if self.return_aux:
            aux: Dict[str, Any] = {
                "gate_alloc": w,   # (B, 2), sums to 1 along dim=-1
                "gate_quality": q, # (B, 1), in (0,1)
            }
            return z, aux
        return z




# class MultiModalFusion(nn.Module):
#     def __init__(self,
#                  num_mod: int,
#                  dim_mod: int,
#                  dim_out: int,
#                  p_drop: float = 0.2,
#     ):
#         super().__init__()
#         self.Wg = nn.Linear(num_mod * dim_mod, num_mod * dim_mod)
#         self.Wz = nn.Linear(num_mod * dim_mod, dim_mod)
#         self.alpha = nn.ParameterList(
#             [nn.Parameter(torch.tensor(1.0)) \
#              for _ in range(num_mod)]
#         )
#         self.mlp = nn.Sequential(
#             nn.LayerNorm(dim_mod), 
#             nn.Linear(dim_mod, dim_mod // 2), 
#             nn.ReLU(), 
#             nn.Dropout(p_drop), 
#             nn.Linear(dim_mod // 2, dim_out)
#         )

#     def forward(self, z: List[torch.Tensor]):
#         z = [alpha_i * z_i for alpha_i, z_i in zip(self.alpha, z)]
#         Z = torch.cat(z, dim=-1)
#         g = torch.sigmoid(self.Wg(Z))
#         fused = g * self.Wz(Z)
#         return self.mlp(fused)
    

class GLRFusion(nn.Module):
    """
    Gated Low Rank Fusion: modality allocation + low-rank bilinear pairwise fusion.

    Args:
      dims_mod:  list of input dims [D1, D2, D3] (one per modality)
      dim_hidden: common hidden dim Dh after per-modality projection
      rank_pair:  low-rank dimension R for pairwise MLB features
      alloc_hidden: hidden width of the tiny allocation MLP (per modality, shared weights)
      dim_out:    output dimension for downstream head
      eps_floor:  epsilon floor mixed with uniform over modalities
      p_drop_mod: probability to drop a modality during training (ModDrop); set 0 to disable
    """
    def __init__(
            self,
            dims_mod: List[int],
            dim_hidden: int   = 128,
            rank_pair: int    = 32,
            alloc_hidden: int = 64,
            dim_out: int      = 512,
            eps_floor: float  = 0.05,
            p_drop_mod: float = 0.0,
            return_aux: bool = False,
    ):
        super().__init__()
        self.M = len(dims_mod)
        self.Dh = dim_hidden
        self.R = rank_pair
        self.eps = eps_floor
        self.p_drop_mod = p_drop_mod
        self.return_aux = return_aux

        # Per-modality pre-norm + projection to common space
        self.pre_ln = nn.ModuleList([nn.LayerNorm(d) for d in dims_mod])
        self.proj   = nn.ModuleList([nn.Linear(d, dim_hidden, bias=False) for d in dims_mod])

        # Shared tiny allocation head
        self.alloc = nn.Sequential(
            nn.Linear(dim_hidden, alloc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(alloc_hidden, 1)
        )

        # Low-rank bilinear projection per modality
        # phi_ij = (A_i h_i) ⊙ (A_j h_j), A_i: Dh -> R (no bias)
        self.A = nn.ModuleList([nn.Linear(dim_hidden, rank_pair, bias=False) for _ in range(self.M)])
        self.pairs = [(i, j) for i in range(self.M) for j in range(i+1, self.M)]
        self.W_pair = nn.Linear(rank_pair, dim_hidden, bias=False)  # shared for sum over pairs

        # Scale for pairwise term
        self.beta = nn.Parameter(torch.tensor(1.0))

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.SiLU(inplace=True),
            nn.Linear(dim_hidden // 2, dim_out)
        )

    def _masked_softmax(self, logits: torch.Tensor, avail: Optional[torch.Tensor]) -> torch.Tensor:
        # logits: (B, M), avail: (B, M) in {0,1} or None
        if avail is not None:
            # large negative where unavailable
            logits = logits.masked_fill(avail == 0, float('-inf'))
        w = F.softmax(logits, dim=-1)
        # if all are masked in a row (rare), fall back to uniform
        nan_rows = torch.isnan(w).any(dim=1)
        if nan_rows.any():
            w[nan_rows] = 1.0 / w.shape[1]
        return w

    def forward(self,
                z_list: List[torch.Tensor],                 # list of (B, D_i)
                avail: Optional[torch.Tensor] = None,       # (B, M) in {0,1}, optional
                force_moddrop: bool = False) -> Tuple[torch.Tensor, dict]:
        assert len(z_list) == self.M
        B = z_list[0].shape[0]
        device = z_list[0].device

        # Optional ModDrop (training-time stochastic dropping)
        if self.training and (self.p_drop_mod > 0 or force_moddrop):
            p = self.p_drop_mod if not force_moddrop else 1.0
            drop = (torch.rand(B, self.M, device=device) < p).float()
            # Ensure at least one modality per sample is kept
            all_dropped = drop.sum(dim=1) == self.M
            if all_dropped.any():
                # randomly un-drop one modality per affected sample
                idx = torch.randint(low=0, high=self.M, size=(int(all_dropped.sum().item()),), device=device)
                drop[all_dropped, idx] = 0.0
            drop_avail = 1.0 - drop
            avail = drop_avail if avail is None else (avail * drop_avail)

        # 1) per-modality pre-norm + projection
        h_list = [proj(ln(z)) for z, ln, proj in zip(z_list, self.pre_ln, self.proj)]  # [(B, Dh)]*M
        H = torch.stack(h_list, dim=1)                                                 # (B, M, Dh)

        # 2) allocation logits per modality (shared tiny MLP)
        alloc_logits = torch.stack([self.alloc(h).squeeze(-1) for h in h_list], dim=1) # (B, M)
        w = self._masked_softmax(alloc_logits, avail)                                  # (B, M)
        # epsilon mix with uniform to avoid dead paths; sums to 1
        g = (1.0 - self.eps) * w + self.eps / self.M                                   # (B, M)

        # 3) low-rank bilinear pairwise interactions (sum over pairs)
        pair_acc = 0.0
        if len(self.pairs) > 0 and self.R > 0:
            for (i, j) in self.pairs:
                ai = self.A[i](h_list[i])    # (B, R)
                aj = self.A[j](h_list[j])    # (B, R)
                pair_acc = pair_acc + (ai * aj)  # (B, R)
            pair_hidden = self.W_pair(pair_acc)  # (B, Dh)
        else:
            pair_hidden = torch.zeros(B, self.Dh, device=device)

        # 4) fuse weighted sum + pairwise
        sum_hidden = torch.sum(g.unsqueeze(-1) * H, dim=1)             # (B, Dh)
        u = F.layer_norm(sum_hidden + self.beta * pair_hidden, (self.Dh,))

        # 5) head
        out = self.head(u)                                             # (B, dim_out)

        if self.return_aux:
            aux: Dict[str, Any] = {
                "alloc": w,           # before epsilon mix, (B, M)
                "gate":  g,           # after epsilon mix, (B, M)
                "pair_scale": float(self.beta.detach().cpu())
            }
            return out, aux
        return out
