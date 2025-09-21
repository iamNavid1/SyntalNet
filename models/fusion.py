import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import Tuple, Optional, Dict, Any, List, Sequence

from models.partial import PartialConv2d, PartialAvgPool2d, PartialGeM
from models.encoder import FMixLowRank
from models.utils import MLP, ChannelLayerNorm2d

class DepthwiseSeparable(nn.Module):
    """
    DW 5x1 + PW 1x1 
    """
    def __init__(self, in_ch: int, out_ch: int, rank: Optional[int] = None):
        super().__init__()
        self.dw = PartialConv2d(in_ch, in_ch, kernel_size=(5,1), padding=(2,0), groups=in_ch, bias=False, return_mask=False)
        if rank is None:
            self.pw = PartialConv2d(in_ch, out_ch, kernel_size=1, bias=False, return_mask=True)
        else:
            self.pw = LowRankPW(in_ch, out_ch, rank=rank) 

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m_grp = m.expand(-1, x.shape[1], -1, -1) if m.shape[1] == 1 else m
        x = self.dw(x, m_grp)
        x, m = self.pw(x, m)
        return x, m


class LowRankPW(nn.Module):
    """
    Low-rank 1x1: in_ch -> rank -> out_ch
    """
    def __init__(self, in_ch: int, out_ch: int, rank: int,
                 act: nn.Module = nn.GELU, p_drop: float = 0.05):
        super().__init__()
        assert rank > 0 and rank <= max(in_ch, out_ch), "rank must be in (0, max(in_ch,out_ch)]"

        self.pw_in  = PartialConv2d(in_ch, rank, kernel_size=1, bias=False, return_mask=True)
        self.act    = act()
        self.drop   = nn.Dropout(p_drop)
        self.pw_out = PartialConv2d(rank, out_ch, kernel_size=1, bias=False, return_mask=True)

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        x, m  = self.pw_in(x, m)
        x  = self.act(x)
        x  = self.drop(x)
        x, m = self.pw_out(x, m)
        return x, m


class ResidualPreNorm(nn.Module):
    """
    Pre-norm residual: LN -> Act. -> optional scaling -> body -> Dropout -> add skip.
    """
    def __init__(
            self,
            in_ch: int, 
            out_ch: int, 
            body: nn.Module, 
            p_drop: float = 0.1, 
            mask_aware_skip: bool = True,
            skip_stride: int | tuple[int,int] = 1,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.ln    = ChannelLayerNorm2d(in_ch)
        self.act   = nn.GELU()
        self.body  = body
        self.drop  = nn.Dropout2d(p_drop)

        self.skip_stride = _pair(skip_stride)
        if not mask_aware_skip:  # vanilla skip
            self._skip_mode = "vanilla"
            self.skip = (
                nn.Identity() if (in_ch == out_ch and self.skip_stride == (1,1)) \
                else nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=self.skip_stride, groups=math.gcd(in_ch, out_ch), bias=False)
            )
        else:  # mask-aware skip
            self._skip_mode = "partial_identity" if (in_ch == out_ch and self.skip_stride == (1,1)) else "partial_1x1"
            need_mask_update = (self.skip_stride != (1,1))
            self.skip = (
                None if (in_ch == out_ch and self.skip_stride == (1,1)) \
                else PartialConv2d(self.in_ch, self.out_ch, kernel_size=1, stride=self.skip_stride, groups=math.gcd(in_ch, out_ch), bias=False, return_mask=need_mask_update)
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
            p_drop_mod_s: Optional[float | Sequence[float]] = None,
            p_drop_mod_e: Optional[float | Sequence[float]] = None,
            warmup_frac: float = 0.4,
            anneal_frac: float = 0.8,
    ):
        super().__init__()
        assert 0.0 < eps < 0.5, "eps should be a small floor < 0.5"
        self.eps = eps
        self.pool = PartialGeM()
        self.head = nn.Linear(2 * C_mod, 3, bias=True)  # 2 for alloc, 1 for quality

        self.M = 2
        if p_drop_mod_s is None:
            self._use_moddrop = False
            p0 = [0.0, 0.0]
            p1 = p0
        else:
            self._use_moddrop = True
            if isinstance(p_drop_mod_s, (float)):
                p0 = [float(p_drop_mod_s)] * self.M
            else:
                assert len(p_drop_mod_s) == self.M
                p0 = list(map(float, p_drop_mod_s))
            if p_drop_mod_e is None:
                p1 = p0
            else:
                if isinstance(p_drop_mod_e, (float)):
                    p1 = [float(p_drop_mod_e)] * self.M
                else:
                    assert len(p_drop_mod_e) == self.M
                    p1 = list(map(float, p_drop_mod_e))

        self.register_buffer("_p_start", torch.tensor(p0, dtype=torch.float32))
        self.register_buffer("_p_end",   torch.tensor(p1, dtype=torch.float32))
        self.warmup_frac = float(warmup_frac)
        self.anneal_frac = float(anneal_frac)
        self._total_epochs = 0

        # Initialize biases
        with torch.no_grad():
            b = self.head.bias
            # [alloc_kp, alloc_vid, qual]
            b.zero_()
            if init_alloc_bias != 0.0:
                b[0] =  init_alloc_bias
                b[1] = -init_alloc_bias
            b[2] = math.log(init_quality / (1.0 - init_quality))

        self._last_aux: Dict[str, Any] = {
            "alloc": None, "quality": None, "drop_rates": self._p_start.clone(),
        }

    def set_total_epochs(self, epochs: int) -> None:
        self._total_epochs = int(epochs)

    def _current_drop_rates(self, epoch: Optional[int]) -> torch.Tensor:
        if (not self.training) or (not self._use_moddrop) or (self._total_epochs == 0) or (epoch is None):
            return self._p_start
        t = epoch / max(1, self._total_epochs)
        if t <= self.warmup_frac:
            return self._p_start
        if t >= self.anneal_frac:
            return self._p_end
        s = (t - self.warmup_frac) / max(1e-6, (self.anneal_frac - self.warmup_frac))
        return self._p_start + s * (self._p_end - self._p_start)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, avail: Optional[torch.Tensor]) -> torch.Tensor:
        # logits: (B, 2), avail: (B, 2) in {0,1} or None
        if avail is not None:
            logits = logits.masked_fill(avail == 0, float("-inf"))
        w = F.softmax(logits, dim=-1)
        # if all are masked in a row (rare), fall back to uniform
        nan_rows = torch.isnan(w).any(dim=1)
        if nan_rows.any():
            w[nan_rows] = 1.0 / w.shape[1]
        return w

    def forward(
            self,
            z_1: torch.Tensor, z_2: torch.Tensor, 
            m_1: torch.Tensor, m_2: torch.Tensor,
            epoch: Optional[int] = None,
            avail: Optional[torch.Tensor] = None,
            force_moddrop: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # s_*: (B, C_mod)
        s_1  = self.pool(z_1, m_1).flatten(1)
        s_2 = self.pool(z_2, m_2).flatten(1)
        s = torch.cat([s_1, s_2], dim=1)   # (B, 2*C_mod)

        logits = self.head(s)                 # (B, 3)
        alloc_logits = logits[:, :2]          # (B, 2)
        qual_logit   = logits[:, 2:3]         # (B, 1)

        B = s.shape[0]
        device = s.device

        if self.training and (self._use_moddrop or force_moddrop):
            if force_moddrop:
                p_vec = torch.ones(self.M, device=device)
            else:
                p_vec = self._current_drop_rates(epoch).to(device)

            drop = torch.bernoulli(p_vec.expand(B, -1))  # (B,2)
            all_dropped = (drop.sum(dim=1) == self.M)
            if all_dropped.any():
                idx = torch.randint(0, self.M, (int(all_dropped.sum().item()),), device=device)
                drop[all_dropped, idx] = 0.0
            drop_mask = 1.0 - drop

            avail = drop_mask if avail is None else (avail.to(device).float() * drop_mask)

        w = F.softmax(alloc_logits, dim=-1)   # (B, 2)
        q = torch.sigmoid(qual_logit)         # (B, 1)

        if avail is None:
            u_avail = torch.full_like(w, 1.0 / self.M)
        else:
            u_avail = avail / (avail.sum(dim=1, keepdim=True) + 1e-8)

        g =  (1.0 - self.eps) * (q * w) + self.eps * u_avail  # (B,2)

        g_1  = g[:, 0:1].view(-1, 1, 1, 1)         # (B,1,1,1)
        g_2  = g[:, 1:2].view(-1, 1, 1, 1)         # (B,1,1,1)

        self._last_aux = {
            "alloc": w.detach(),             # (B,2)
            "quality": q.detach(),           # (B,1)
            "drop_rates": self._current_drop_rates(epoch).detach().cpu(),
        }

        return g_1, g_2

    def get_aux(self) -> Dict[str, Any]:
        return self._last_aux


class GPSFusion(nn.Module):
    """
    Gated Partial Separable Fusion:
    Inputs:  z_kp, z_emb  each (B, C_mod, T, F)
    Pipeline:
      - LN per modality
      - Two-factor gate (quality * allocation) from normalized streams
      - Concat on C
      - ResidualPreNorm (apply gating)
      - AvgPool2d (1,2) on F
      - ResidualPreNorm (DW+PW)
      - GeM -> Dropout -> Linear head

    Returns:
      logits (B, out_dim)
    """
    def __init__(self,
                C_mod: int,
                F_mod: int,
                pool_schedule: List[int | Tuple[int, int]],
                channel_schedule: List[int],
                target_grid: Tuple[int,int] = (4, 4),
                pw_rank: Optional[int] = None,
                p_drop_res: float = 0.05,
                p_drop_head: float = 0.05,
                gate_eps: float = 0.05,
                gate_init_quality: float = 0.8,
                gate_init_alloc_bias: float = 0.0,
                p_drop_mod_s: Optional[float | Sequence[float]] = None,
                p_drop_mod_e: Optional[float | Sequence[float]] = None,
                warmup_frac: float = 0.4,
                anneal_frac: float = 0.8,
    ):
        super().__init__()
        self.C_mod = C_mod
        self.target_grid = tuple(target_grid)

        # ---- derive channel schedule & final C ----
        self.channel_schedule = list(channel_schedule)
        Ht, Wf = self.target_grid
        self.out_dim = self.channel_schedule[-1] * Ht * Wf

        # ---- pre-norm per modality ----
        self.ln_1 = ChannelLayerNorm2d(self.C_mod)
        self.ln_2 = ChannelLayerNorm2d(self.C_mod)

        # ---- two-factor modality gate ----
        self.mod_gate = ModalityGate(
            C_mod=self.C_mod,
            eps=gate_eps,
            init_quality=gate_init_quality,
            init_alloc_bias=gate_init_alloc_bias,
            p_drop_mod_s=p_drop_mod_s,
            p_drop_mod_e=p_drop_mod_e,
            warmup_frac=warmup_frac,
            anneal_frac=anneal_frac,
        )

        # ---- residual blocks ----
        blocks: List[ResidualPreNorm] = []
        in_c = self.C_mod

        for out_c in self.channel_schedule:
            body = DepthwiseSeparable(in_c, out_c, pw_rank)
            blocks.append(
                ResidualPreNorm(
                    in_ch=in_c, out_ch=out_c,
                    body=body,
                    p_drop=p_drop_res,
                    mask_aware_skip=True,
                )
            )
            in_c = out_c

        self.blocks = nn.ModuleList(blocks)

        # ---- downsampling schedule ----
        temporal_pools: List[PartialGeM] = []
        feature_pools: List[FMixLowRank] = []
        cur_F = F_mod

        for pool in pool_schedule:
            p_t, p_f = _pair(pool)
            temporal_pools.append(
                PartialGeM((p_t, 1), global_pool=False, return_mask=True)
            )
            feature_pools.append(
                FMixLowRank(F_in=cur_F, F_out=cur_F//p_f, rank=max(8, cur_F//4), act=nn.GELU, pre_norm=True, bias=True)
            )
            cur_F //= p_f

        self.temporal_pools = nn.ModuleList(temporal_pools)
        self.feature_pools = nn.ModuleList(feature_pools)

        # ---- guardrail to ensure target grid output dimension ----
        self.adaptive = nn.AdaptiveAvgPool2d(self.target_grid)

        # ---- final head dropout ----
        self.drop = nn.Dropout(p_drop_head)

    def set_total_epochs(self, epochs: int) -> None:
        self.mod_gate.set_total_epochs(epochs)

    def get_aux(self) -> Dict[str, Any]:
        return self.mod_gate._last_aux

    @staticmethod
    def _or_masks(m1: torch.Tensor, m2: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        # both are (B,1,T,F) in your code
        if m1.shape[1] != 1:
            m1 = (m1.sum(dim=1, keepdim=True) > 0).to(dtype)
        if m2.shape[1] != 1:
            m2 = (m2.sum(dim=1, keepdim=True) > 0).to(dtype)
        return torch.maximum(m1.to(dtype), m2.to(dtype))

    def forward(
            self,
            z_1: torch.Tensor, z_2: torch.Tensor, 
            m_1: torch.Tensor, m_2: torch.Tensor,
            epoch: Optional[int] = None,
            avail: Optional[torch.Tensor] = None,
            force_moddrop: bool = False
    ) -> torch.Tensor:
        # normalize each modality first for stable gate statistics
        z_1 = self.ln_1(z_1)    # (B, C_mod, T, F)
        z_2 = self.ln_2(z_2)     # (B, C_mod, T, F)

        m = self._or_masks(m_1, m_2, dtype=z_1.dtype)

        # two-factor gate
        g_1, g_2 = self.mod_gate(
            z_1, z_2, m_1, m_2,
            epoch, avail, force_moddrop,
        )

        # fuse
        z = g_1 * z_1 + g_2 * z_2                    # (B, C_mod, T, F)

        for i, block in enumerate(self.blocks):
            z, m = block(z, m)
            if i < len(self.temporal_pools):
                z, m = self.temporal_pools[i](z, m)  # early downsampling
                z, m = self.feature_pools[i](z, m)   # early downsampling

        # adaptive pooling for guardrail
        z = self.adaptive(z)  # (B, C_final, Ht, Wf)

        # final head
        z = z.flatten(1)      # (B, C_final*Ht*Wf) == self.out_dim
        z = self.drop(z)
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

    :param dims_mod:         list of input dims [D1, D2, D3] (one per modality)
    :param dim_hidden:       common hidden dim Dh after per-modality projection
    :param rank_pair:        low-rank dimension R for pairwise MLB features
    :param alloc_hidden:     hidden width of the tiny allocation MLP (per modality, shared weights)
    :param dim_out:          output dimension for downstream head
    :param eps_floor:        epsilon floor mixed with uniform over modalities
    :param p_drop_mod:       start drop prob (scalar or sequence length M); set None to disable
    :param p_drop_mod_final: optional end drop prob (sequence length M) for linear anneal; if None uses p_drop_mod
    :param warmup_frac:      keep start rates until this fraction of training
    :param anneal_frac:      finish anneal by this fraction of training
    :param drop_mode:        independent "Bernoulli" masking or "single" to drop at most one modality
    """
    def __init__(
            self,
            dims_mod: List[int],
            dim_hidden: int   = None,
            rank_pair: int    = 32,
            alloc_hidden: int = 64,
            dim_out: int      = 256,
            eps_floor: float  = 0.05,
            p_drop: float     = 0.1,
            p_drop_mod_s:      Optional[float | Sequence[float]] = None,
            p_drop_mod_e:      Optional[float | Sequence[float]] = None,
            warmup_frac: float = 0.4,
            anneal_frac: float = 0.8,
            drop_mode: str     = "bernoulli",  # or "single"
    ):
        super().__init__()
        self.M = len(dims_mod)
        self.R = rank_pair
        self.eps = eps_floor
        self.drop = nn.Dropout(p_drop)
        self.drop_mode = drop_mode

        if dim_hidden is None:
            if all(d == dims_mod[0] for d in dims_mod):
                dim_hidden = dims_mod[0]
            else:
                raise ValueError("dim_hidden must be specified if input dims differ")
        self.Dh = dim_hidden

        if p_drop_mod_s is None or all(x is None for x in p_drop_mod_s):
            self._use_moddrop = False
            p0 = [0.0] * self.M
            p1 = p0
        else:
            self._use_moddrop = True
            if isinstance(p_drop_mod_s, (float, int)):
                p0 = [float(p_drop_mod_s)] * self.M
            else:
                assert len(p_drop_mod_s) == self.M
                p0 = list(map(float, p_drop_mod_s))
            if p_drop_mod_e is None:
                p1 = p0
            else:
                if isinstance(p_drop_mod_e, (float, int)):
                    p1 = [float(p_drop_mod_e)] * self.M
                else:
                    assert len(p_drop_mod_e) == self.M
                    p1 = list(map(float, p_drop_mod_e))

        self.register_buffer("_p_start", torch.tensor(p0, dtype=torch.float32))
        self.register_buffer("_p_end",   torch.tensor(p1, dtype=torch.float32))
        self.warmup_frac   = float(warmup_frac)
        self.anneal_frac   = float(anneal_frac)
        self._total_epochs = 0

        # Per-modality pre-norm + projection to common space
        self.pre_ln = nn.ModuleList([nn.LayerNorm(d) for d in dims_mod])
        self.proj = nn.ModuleList([
            nn.Linear(d, dim_hidden, bias=False) if d != dim_hidden else nn.Identity()
            for d in dims_mod
        ])

        # Shared allocation head
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
        self.beta = nn.Parameter(torch.tensor(1.0))  # scale for pairwise term

        # Output head
        self.head = MLP(in_features=dim_hidden, out_features=dim_out)

        self._last_aux: Dict[str, Any] = {
            "alloc": None, "gate": None, "pair_scale": float(1.0), "drop_rates": self._p_start.clone()
        }

    def set_total_epochs(self, epochs: int) -> None:
        self._total_epochs = int(epochs)

    def _current_drop_rates(self, epoch) -> torch.Tensor:
        if not self.training or not self._use_moddrop or self._total_epochs==0:
            return self._p_start
        t = epoch / max(1, self._total_epochs)
        if t <= self.warmup_frac:
            return self._p_start
        if t >= self.anneal_frac:
            return self._p_end
        s = (t - self.warmup_frac) / max(1e-6, (self.anneal_frac - self.warmup_frac))
        return self._p_start + s * (self._p_end - self._p_start)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, avail: Optional[torch.Tensor]) -> torch.Tensor:
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
                epoch: Optional[int] = None,
                avail: Optional[torch.Tensor] = None,       # (B, M) in {0,1}, optional
                force_moddrop: bool = False
    ) -> torch.Tensor:
        assert len(z_list) == self.M
        B = z_list[0].shape[0]
        device = z_list[0].device

        # Optional ModDrop (training-time stochastic dropping)
        if self.training and (self._use_moddrop or force_moddrop):
            if force_moddrop:
                p_vec = torch.ones(self.M, device=device)
            else:
                p_vec = self._current_drop_rates(epoch).to(device)
            
            if self.drop_mode == "single":
                # Drop at most one modality
                rho = float(p_vec.sum().clamp(max=1.0).item())
                drop_mask = torch.ones(B, self.M, device=device)
                if rho > 0:
                    drop_flag = (torch.rand(B, device=device) < rho)
                    if drop_flag.any():
                        probs = (p_vec / p_vec.sum()).clamp_min(1e-8)
                        idx = torch.multinomial(probs, int(drop_flag.sum().item()), replacement=True).to(device)
                        drop_mask[drop_flag, idx] = 0.0
            else:
                # Independent Bernoulli per modality
                drop = torch.bernoulli(p_vec.expand(B, -1))  # (B,M)
                all_dropped = (drop.sum(dim=1) == self.M)
                if all_dropped.any():  # ensure ≥1 kept
                    idx = torch.randint(0, self.M, (int(all_dropped.sum().item()),), device=device)
                    drop[all_dropped, idx] = 0.0
                drop_mask = 1.0 - drop

            avail = drop_mask if avail is None else (avail.to(device).float() * drop_mask)

        # 1) per-modality pre-norm + projection
        h_list = [proj(ln(z)) for z, ln, proj in zip(z_list, self.pre_ln, self.proj)]  # [(B, Dh)]*M
        H = torch.stack(h_list, dim=1)                                                 # (B, M, Dh)

        # 2) allocation logits per modality (shared tiny MLP)
        alloc_logits = torch.stack([self.alloc(h).squeeze(-1) for h in h_list], dim=1) # (B, M)
        w = self._masked_softmax(alloc_logits, avail)  # sums to 1 over active
        if avail is None:
            u_avail = torch.full_like(w, 1.0 / self.M)
        else:
            u_avail = avail / (avail.sum(dim=1, keepdim=True) + 1e-8)
        # epsilon mix with uniform to avoid dead paths; masked modalities stay 0
        g = (1.0 - self.eps) * w + self.eps * u_avail  # (B, M)

        # 3) low-rank bilinear pairwise interactions (sum over pairs)
        if len(self.pairs) > 0 and self.R > 0:
            pair_acc = 0.0
            for (i, j) in self.pairs:
                ai = self.A[i](h_list[i])  # (B, R)
                aj = self.A[j](h_list[j])  # (B, R)
                if avail is not None:
                    m_ij = (avail[:, i] * avail[:, j]).unsqueeze(-1)
                    pair_acc += m_ij * (ai * aj)
                else:
                    pair_acc += (ai * aj)
            if avail is not None:
                n_active = avail.sum(dim=1)
                n_pairs = torch.clamp(n_active * (n_active - 1) / 2, min=1.0).unsqueeze(-1)
                full_pairs = float(self.M * (self.M - 1) / 2) or 1.0
                pair_acc = pair_acc * (full_pairs / n_pairs)
            pair_hidden = self.W_pair(pair_acc)  # (B, Dh)
        else:
            pair_hidden = torch.zeros(B, self.Dh, device=device)

        # 4) fuse weighted sum + pairwise
        sum_hidden = torch.sum(g.unsqueeze(-1) * H, dim=1)  # (B, Dh)
        u = F.layer_norm(sum_hidden + self.beta * pair_hidden, (self.Dh,))
        u = self.drop(u)

        # 5) head
        out = self.head(u)  # (B, dim_out)

        self._last_aux = {
            "alloc": w.detach(),  # before epsilon mix, (B, M)
            "gate":  g.detach(),  # after epsilon mix, (B, M)
            "pair_scale": float(self.beta.detach().cpu()),
            "drop_rates": self._current_drop_rates(epoch).detach().cpu(),
        }

        return out

    def get_aux(self) -> Dict[str, Any]:
        return self._last_aux




# def gate_entropy_regularizer(gate_alloc: torch.Tensor, strength: float = 1e-4) -> torch.Tensor:
#     if strength <= 0:
#         return torch.zeros((), device=gate_alloc.device, dtype=gate_alloc.dtype)
#     w = gate_alloc.clamp_min(1e-8)
#     ent = -(w * w.log()).sum(dim=-1).mean()
#     return -strength * ent  # negative because ent is positive

# # If some streams are unavailable (or ModDrop masked), compute entropy over only the available entries:
# def gate_entropy_regularizer_masked(w: torch.Tensor, avail: torch.Tensor | None, strength: float = 1e-4):
#     if strength <= 0: 
#         return torch.zeros((), device=w.device, dtype=w.dtype)
#     if avail is not None:
#         # renormalize w over available entries only
#         w = w * (avail > 0).to(w.dtype)
#         w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
#     w = w.clamp_min(1e-8)
#     ent = -(w * w.log()).sum(dim=-1).mean()
#     return -strength * ent

# def ent_weight(epoch, warm=0, hold=5, decay=10, max_lambda=1e-4):
#     if epoch < warm:         return 0.0
#     if epoch < warm+hold:    return max_lambda
#     t = min(1.0, (epoch - warm - hold) / max(1, decay))
#     return max_lambda * 0.5 * (1 + math.cos(math.pi * t))  # cosine to 0

# λm = ent_weight(epoch, hold=5, decay=5, max_lambda=1e-4)
# loss += gate_entropy_regularizer(aux_mod['gate_alloc'], λm)
# λM = ent_weight(epoch, hold=5, decay=5, max_lambda=2e-4)
# loss += gate_entropy_regularizer_masked(aux_mm['alloc'], avail, λM)
