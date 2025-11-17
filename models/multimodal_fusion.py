import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

from models.partial import PartialGeM, PartialMean2d
from models.encoder import CNXv2Block
from models.utils import ChannelLayerNorm2d


# -----------------------------------------------------------------------------
#                       Gated Low Rank Cross-modal Fusion
# -----------------------------------------------------------------------------
class GLR_X(nn.Module):
    def __init__(
        self,
        num_mod: int,
        dims_mod: int,
        rank_pair: int = 16,
        alloc_hidden: int = 32,
        dim_out: int | None = None,
        eps_floor: float = 0.03,
        p_drop: float = 0.1
    ):
        super().__init__()
        self.M   = num_mod
        self.Dh  = dims_mod
        self.R   = rank_pair
        self.out = dim_out or dims_mod
        self.eps = eps_floor
        self.drop = nn.Dropout(p_drop)

        # per-modality pre-LN
        self.pre_ln = nn.ModuleList([nn.LayerNorm(dims_mod, eps=1e-5) for _ in range(self.M)])

        # allocation head (shared)
        self.alloc = nn.Sequential(
            nn.Linear(dims_mod, alloc_hidden, bias=True),
            nn.GELU(),
            nn.Linear(alloc_hidden, 1, bias=True)
        )
        nn.init.zeros_(self.alloc[-1].weight)
        nn.init.zeros_(self.alloc[-1].bias)

        # low-rank bilinear
        self.A = nn.ModuleList([nn.Linear(dims_mod, rank_pair, bias=False) for _ in range(self.M)])
        self.pairs = [(i, j) for i in range(self.M) for j in range(i+1, self.M)]
        self.W_pair = nn.Linear(rank_pair, dims_mod, bias=False)
        self.beta = nn.Parameter(torch.tensor(-0.848))  # ~30%
        self.gamma_res = nn.Parameter(torch.tensor(-2.945))  # ~5%

        for a in self.A:
            nn.init.xavier_uniform_(a.weight, gain=1.0)
        nn.init.xavier_uniform_(self.W_pair.weight, gain=1.0)

        # residual concat-MLP
        self.res_ln = nn.LayerNorm(self.M * dims_mod, eps=1e-5)
        self.res_mlp = nn.Sequential(
            nn.Linear(self.M * dims_mod, dims_mod, bias=True),
            nn.GELU(),
            nn.Dropout(p_drop)
        )

        # output head
        self.head = nn.Sequential(
            nn.LayerNorm(dims_mod, eps=1e-5),
            nn.Linear(dims_mod, self.out, bias=True)
        )

        self._last_aux = None

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, avail: Optional[torch.Tensor]) -> torch.Tensor:
        if avail is not None:
            logits = logits.masked_fill(avail == 0, float('-inf'))
        w = F.softmax(logits, dim=-1)
        bad = torch.isnan(w).any(dim=1)
        if bad.any():
            w[bad] = 1.0 / w.shape[1]
        return w

    def get_aux(self):
        return self._last_aux

    def forward(self, z_list: list[torch.Tensor],
                avail: Optional[torch.Tensor] = None,
                tau: float = 1.0) -> torch.Tensor:
        # z_i: (B, Dh), avail: (B, M) in {0,1}
        B = z_list[0].shape[0]
        device = z_list[0].device

        # 1) pre-LN
        h_list = [ln(z) for z, ln in zip(z_list, self.pre_ln)]  # [(B,Dh)]*M
        H = torch.stack(h_list, dim=1)                           # (B,M,Dh)

        # 2) gates (softmax + epsilon floor)
        alloc_logits = torch.stack([self.alloc(h).squeeze(-1) for h in h_list], dim=1)  # (B,M)
        w = self._masked_softmax(alloc_logits / max(tau, 1e-3), avail)
        if avail is None:
            u_avail = torch.full_like(w, 1.0 / self.M)
        else:
            u_avail = avail / (avail.sum(dim=1, keepdim=True) + 1e-8)
        g = (1.0 - self.eps) * w + self.eps * u_avail           # (B,M)

        # 3) low-rank pairwise
        pair_acc = torch.zeros(B, self.R, device=device, dtype=H.dtype)
        for (i, j) in self.pairs:
            ai = self.A[i](h_list[i])  # (B,R)
            aj = self.A[j](h_list[j])  # (B,R)
            term = ai * aj
            if avail is not None:
                term = term * (avail[:, i] * avail[:, j]).unsqueeze(-1)
            term = term * (g[:, i] * g[:, j]).unsqueeze(-1)
            pair_acc = pair_acc + term
        if avail is not None:
            n_active = avail.sum(dim=1)
            n_pairs = torch.clamp(n_active * (n_active - 1) / 2, min=1.0).unsqueeze(-1)
            full_pairs = float(self.M * (self.M - 1) / 2) or 1.0
            pair_acc = pair_acc * (full_pairs / n_pairs)
        pair_hidden = self.W_pair(pair_acc)  # (B,Dh)

        # 4) weighted sum + residual concat-MLP
        sum_hidden = (g.unsqueeze(-1) * H).sum(dim=1)           # (B,Dh)
        sum_hidden = F.layer_norm(sum_hidden, (self.Dh,))
        pair_hidden = F.layer_norm(pair_hidden, (self.Dh,))
        beta = torch.sigmoid(self.beta)
        core = (1 - beta) * sum_hidden + beta * pair_hidden
        r = self.res_mlp(self.res_ln(torch.cat(z_list, dim=-1)))  # (B,Dh)
        gamma = torch.sigmoid(self.gamma_res)
        u = self.drop(core + gamma * r)

        with torch.no_grad():
            pair_mag = pair_hidden.norm(dim=1).mean()
            sum_mag  = sum_hidden.norm(dim=1).mean()
            res_mag  = r.norm(dim=1).mean()
            head_mag = u.norm(dim=1).mean()
            self._last_aux = {
                "alloc": w.detach().cpu(),              # (B,M)
                "gate":  g.detach().cpu(),              # (B,M)
                "pair_mag": pair_mag.detach().cpu(),
                "sum_mag":  sum_mag.detach().cpu(),
                "res_mag":  res_mag.detach().cpu(),
                "head_mag": head_mag.detach().cpu(),
                "beta": float(beta.detach().cpu()),
                "gamma_res": float(gamma.detach().cpu()),
            }

        return self.head(u)


# -----------------------------------------------------------------------------
#                         Baseline: UniformAvgFusion
# -----------------------------------------------------------------------------
class UniformAvgFusion(nn.Module):
    """
    Baseline: Simple mean-pooling across modalities.
    """
    def __init__(
        self,
        num_mod: int,
        dims_mod: int,
        dim_out: Optional[int] = None,
        use_ln: bool = True,
        **kwargs
    ):
        super().__init__()
        self.M = num_mod
        self.D = dims_mod
        self.out = dim_out or dims_mod
        self.use_ln = use_ln
        if use_ln:
            self.pre_ln = nn.ModuleList([
                nn.LayerNorm(dims_mod, eps=1e-5) for _ in range(self.M)
            ])
        else:
            self.pre_ln = None
        self.head = nn.Linear(dims_mod, self.out, bias=True)
        self._last_aux = None

    def get_aux(self):
        return self._last_aux

    def forward(self,
                z_list: List[torch.Tensor],
                avail: Optional[torch.Tensor] = None,
                tau: float = 1.0) -> torch.Tensor:
        assert len(z_list) == self.M
        B = z_list[0].shape[0]
        device = z_list[0].device

        if self.use_ln:
            h_list = [ln(z) for z, ln in zip(z_list, self.pre_ln)]
        else:
            h_list = z_list

        H = torch.stack(h_list, dim=1)  # (B,M,D)

        if avail is None:
            w = torch.full((B, self.M), 1.0 / self.M,
                           device=device, dtype=H.dtype)
        else:
            denom = avail.sum(dim=1, keepdim=True).clamp_min(1.0)
            w = avail / denom

        fused = (w.unsqueeze(-1) * H).sum(dim=1)  # (B,D)
        
        with torch.no_grad():
            self._last_aux = {
                "weights": w.detach().cpu(),
                "fused_mag": fused.norm(dim=1).mean().detach().cpu(),
            }
        
        return self.head(fused)
# -------------------------------------------------------------------------
#              Baseline: Gated Sum (No Pairwise Term)
# -------------------------------------------------------------------------
class GatedSumOnly(nn.Module):
    """
    GLR without low-rank pairwise term:
      - pre-LN
      - allocation head + eps-floor gating
      - weighted sum of modalities
      - optional residual concat-MLP
    """
    def __init__(
        self,
        num_mod: int,
        dims_mod: int,
        alloc_hidden: int = 32,
        dim_out: int | None = None,
        eps_floor: float = 0.03,
        p_drop: float = 0.1,
        use_residual: bool = True,
        **kwargs
    ):
        super().__init__()
        self.M   = num_mod
        self.Dh  = dims_mod
        self.out = dim_out or dims_mod
        self.eps = eps_floor
        self.drop = nn.Dropout(p_drop)
        self.use_residual = use_residual

        self.pre_ln = nn.ModuleList([
            nn.LayerNorm(dims_mod, eps=1e-5) for _ in range(self.M)
        ])

        self.alloc = nn.Sequential(
            nn.Linear(dims_mod, alloc_hidden, bias=True),
            nn.GELU(),
            nn.Linear(alloc_hidden, 1, bias=True)
        )
        nn.init.zeros_(self.alloc[-1].weight)
        nn.init.zeros_(self.alloc[-1].bias)

        if use_residual:
            self.res_ln = nn.LayerNorm(self.M * dims_mod, eps=1e-5)
            self.res_mlp = nn.Sequential(
                nn.Linear(self.M * dims_mod, dims_mod, bias=True),
                nn.GELU(),
                nn.Dropout(p_drop)
            )
            self.gamma_res = nn.Parameter(torch.tensor(-2.945))
        else:
            self.res_ln = None
            self.res_mlp = None
            self.gamma_res = None

        self.head = nn.Sequential(
            nn.LayerNorm(dims_mod, eps=1e-5),
            nn.Linear(dims_mod, self.out, bias=True)
        )

        self._last_aux: Optional[Dict] = None

    @staticmethod
    def _masked_softmax(logits: torch.Tensor,
                        avail: Optional[torch.Tensor]) -> torch.Tensor:
        if avail is not None:
            logits = logits.masked_fill(avail == 0, float('-inf'))
        w = F.softmax(logits, dim=-1)
        bad = torch.isnan(w).any(dim=1)
        if bad.any():
            w[bad] = 1.0 / w.shape[1]
        return w

    def get_aux(self):
        return self._last_aux

    def forward(self,
                z_list: List[torch.Tensor],
                avail: Optional[torch.Tensor] = None,
                tau: float = 1.0) -> torch.Tensor:
        assert len(z_list) == self.M
        B = z_list[0].shape[0]
        device = z_list[0].device

        h_list = [ln(z) for z, ln in zip(z_list, self.pre_ln)]
        H = torch.stack(h_list, dim=1)  # (B,M,D)

        alloc_logits = torch.stack(
            [self.alloc(h).squeeze(-1) for h in h_list], dim=1
        )  # (B,M)
        w = self._masked_softmax(alloc_logits / max(tau, 1e-3), avail)
        if avail is None:
            u_avail = torch.full_like(w, 1.0 / self.M)
        else:
            u_avail = avail / (avail.sum(dim=1, keepdim=True) + 1e-8)
        g = (1.0 - self.eps) * w + self.eps * u_avail  # (B,M)

        sum_hidden = (g.unsqueeze(-1) * H).sum(dim=1)   # (B,D)
        sum_hidden = F.layer_norm(sum_hidden, (self.Dh,))

        if self.use_residual:
            r = self.res_mlp(self.res_ln(torch.cat(z_list, dim=-1)))  # (B,D)
            gamma = torch.sigmoid(self.gamma_res)
            u = self.drop(sum_hidden + gamma * r)
        else:
            u = self.drop(sum_hidden)

        with torch.no_grad():
            sum_mag = sum_hidden.norm(dim=1).mean()
            self._last_aux = {
                "alloc": w.detach().cpu(),
                "gate":  g.detach().cpu(),
                "sum_mag": sum_mag.detach().cpu(),
            }

        return self.head(u)


# -------------------------------------------------------------------------
#              Baseline: Pairwise Only (No Gated-Sum Pathway)
# -------------------------------------------------------------------------
class PairwiseOnly(nn.Module):
    """
    GLR_X with only pairwise low-rank term; gated-sum pathway removed:
      - pre-LN
      - allocation head + eps-floor gating (for weighting pairs)
      - pairwise low-rank interactions only
      - optional residual concat-MLP
    """
    def __init__(
        self,
        num_mod: int,
        dims_mod: int,
        rank_pair: int = 16,
        alloc_hidden: int = 32,
        dim_out: int | None = None,
        eps_floor: float = 0.03,
        p_drop: float = 0.1,
        use_residual: bool = True,
        **kwargs
    ):
        super().__init__()
        self.M   = num_mod
        self.Dh  = dims_mod
        self.R   = rank_pair
        self.out = dim_out or dims_mod
        self.eps = eps_floor
        self.drop = nn.Dropout(p_drop)
        self.use_residual = use_residual

        # per-modality pre-LN
        self.pre_ln = nn.ModuleList([
            nn.LayerNorm(dims_mod, eps=1e-5) for _ in range(self.M)
        ])

        # allocation head (shared)
        self.alloc = nn.Sequential(
            nn.Linear(dims_mod, alloc_hidden, bias=True),
            nn.GELU(),
            nn.Linear(alloc_hidden, 1, bias=True)
        )
        nn.init.zeros_(self.alloc[-1].weight)
        nn.init.zeros_(self.alloc[-1].bias)

        # low-rank bilinear
        self.A = nn.ModuleList([
            nn.Linear(dims_mod, rank_pair, bias=False) for _ in range(self.M)
        ])
        self.pairs = [(i, j) for i in range(self.M) for j in range(i+1, self.M)]
        self.W_pair = nn.Linear(rank_pair, dims_mod, bias=False)

        for a in self.A:
            nn.init.xavier_uniform_(a.weight, gain=1.0)
        nn.init.xavier_uniform_(self.W_pair.weight, gain=1.0)

        # residual concat-MLP
        if use_residual:
            self.res_ln = nn.LayerNorm(self.M * dims_mod, eps=1e-5)
            self.res_mlp = nn.Sequential(
                nn.Linear(self.M * dims_mod, dims_mod, bias=True),
                nn.GELU(),
                nn.Dropout(p_drop)
            )
            self.gamma_res = nn.Parameter(torch.tensor(-2.945))  # ~5%
        else:
            self.res_ln = None
            self.res_mlp = None
            self.gamma_res = None

        # output head
        self.head = nn.Sequential(
            nn.LayerNorm(dims_mod, eps=1e-5),
            nn.Linear(dims_mod, self.out, bias=True)
        )

        self._last_aux = None

    @staticmethod
    def _masked_softmax(logits: torch.Tensor,
                        avail: Optional[torch.Tensor]) -> torch.Tensor:
        if avail is not None:
            logits = logits.masked_fill(avail == 0, float('-inf'))
        w = F.softmax(logits, dim=-1)
        bad = torch.isnan(w).any(dim=1)
        if bad.any():
            w[bad] = 1.0 / w.shape[1]
        return w

    def get_aux(self):
        return self._last_aux

    def forward(self,
                z_list: List[torch.Tensor],
                avail: Optional[torch.Tensor] = None,
                tau: float = 1.0) -> torch.Tensor:
        assert len(z_list) == self.M
        B = z_list[0].shape[0]
        device = z_list[0].device

        # 1) pre-LN
        h_list = [ln(z) for z, ln in zip(z_list, self.pre_ln)]  # [(B,Dh)]*M
        H = torch.stack(h_list, dim=1)                           # (B,M,Dh)

        # 2) gates (softmax + epsilon floor)
        alloc_logits = torch.stack(
            [self.alloc(h).squeeze(-1) for h in h_list], dim=1
        )  # (B,M)
        w = self._masked_softmax(alloc_logits / max(tau, 1e-3), avail)
        if avail is None:
            u_avail = torch.full_like(w, 1.0 / self.M)
        else:
            u_avail = avail / (avail.sum(dim=1, keepdim=True) + 1e-8)
        g = (1.0 - self.eps) * w + self.eps * u_avail  # (B,M)

        # 3) low-rank pairwise (only pathway)
        pair_acc = torch.zeros(B, self.R, device=device, dtype=H.dtype)
        for (i, j) in self.pairs:
            ai = self.A[i](h_list[i])  # (B,R)
            aj = self.A[j](h_list[j])  # (B,R)
            term = ai * aj
            if avail is not None:
                term = term * (avail[:, i] * avail[:, j]).unsqueeze(-1)
            term = term * (g[:, i] * g[:, j]).unsqueeze(-1)
            pair_acc = pair_acc + term
        if avail is not None:
            n_active = avail.sum(dim=1)
            n_pairs = torch.clamp(n_active * (n_active - 1) / 2, min=1.0).unsqueeze(-1)
            full_pairs = float(self.M * (self.M - 1) / 2) or 1.0
            pair_acc = pair_acc * (full_pairs / n_pairs)
        pair_hidden = self.W_pair(pair_acc)  # (B,Dh)
        pair_hidden = F.layer_norm(pair_hidden, (self.Dh,))

        # 4) residual concat-MLP (optional)
        if self.use_residual:
            r = self.res_mlp(self.res_ln(torch.cat(z_list, dim=-1)))  # (B,Dh)
            gamma = torch.sigmoid(self.gamma_res)
            u = self.drop(pair_hidden + gamma * r)
        else:
            u = self.drop(pair_hidden)

        with torch.no_grad():
            pair_mag = pair_hidden.norm(dim=1).mean()
            self._last_aux = {
                "alloc": w.detach().cpu(),
                "gate":  g.detach().cpu(),
                "pair_mag": pair_mag.detach().cpu(),
            }
            if self.use_residual:
                res_mag = r.norm(dim=1).mean()
                self._last_aux["res_mag"] = res_mag.detach().cpu()
                self._last_aux["gamma_res"] = float(gamma.detach().cpu())

        return self.head(u)


# -------------------------------------------------------------------------
#              Baseline: Concat MLP (No Gating / No Pairwise)
# -------------------------------------------------------------------------
class ConcatMLP(nn.Module):
    """
    Simple late fusion baseline:
      - concatenate all modality vectors
      - LN + MLP
    """
    def __init__(
        self,
        num_mod: int,
        dims_mod: int,
        dim_out: int | None = None,
        hidden: Optional[int] = None,
        p_drop: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.M   = num_mod
        self.Dh  = dims_mod
        self.out = dim_out or dims_mod
        H = hidden or (2 * dims_mod)

        self.ln_in = nn.LayerNorm(self.M * self.Dh, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(self.M * self.Dh, H, bias=True),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(H, self.out, bias=True)
        )
        self._last_aux = None

    def get_aux(self):
        return self._last_aux

    def forward(self,
                z_list: List[torch.Tensor],
                avail: Optional[torch.Tensor] = None,
                tau: float = 1.0) -> torch.Tensor:
        z_cat = torch.cat(z_list, dim=-1)  # (B,M*D)
        z_cat = self.ln_in(z_cat)
        out = self.mlp(z_cat)
        
        with torch.no_grad():
            self._last_aux = {
                "concat_mag": z_cat.norm(dim=1).mean().detach().cpu(),
                "out_mag": out.norm(dim=1).mean().detach().cpu(),
            }
        
        return out

