import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from models.partial import PartialGeM, PartialMean2d
from models.encoder import CNXv2Block
from models.utils import ChannelLayerNorm2d


class BSX(nn.Module):
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
            mean_norm = mean.norm(dim=1).mean()         # scalar
            gem_norm  = gem.norm(dim=1).mean()          # scalar
            vec_norm  = vec.norm(dim=1).mean()          # scalar
            self._last_aux = {
                "mask_cov_overall": cov.detach().cpu(),
                "mean_norm": mean_norm.detach().cpu(),
                "gem_norm":  gem_norm.detach().cpu(),
                "vec_norm":  vec_norm.detach().cpu(),
            }

        return vec


class GLRFusion(nn.Module):
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
        self.beta = nn.Parameter(torch.tensor(0.1))  # softer start

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

        # 3) low-rank pairwise (fixed init, bug-fixed accumulator)
        pair_acc = torch.zeros(B, self.R, device=device, dtype=H.dtype)
        for (i, j) in self.pairs:
            ai = self.A[i](h_list[i])  # (B,R)
            aj = self.A[j](h_list[j])  # (B,R)
            term = ai * aj
            if avail is not None:
                term = term * (avail[:, i] * avail[:, j]).unsqueeze(-1)
            pair_acc = pair_acc + term
        if avail is not None:
            n_active = avail.sum(dim=1)
            n_pairs = torch.clamp(n_active * (n_active - 1) / 2, min=1.0).unsqueeze(-1)
            full_pairs = float(self.M * (self.M - 1) / 2) or 1.0
            pair_acc = pair_acc * (full_pairs / n_pairs)
        pair_hidden = self.W_pair(pair_acc)  # (B,Dh)

        # 4) weighted sum + residual concat-MLP
        sum_hidden = (g.unsqueeze(-1) * H).sum(dim=1)           # (B,Dh)
        core = F.layer_norm(sum_hidden + self.beta * pair_hidden, (self.Dh,))
        r = self.res_mlp(self.res_ln(torch.cat(z_list, dim=-1)))  # (B,Dh)
        u = self.drop(core + r)

        with torch.no_grad():
            # alloc logits softmax (w), after epsilon-mix (g), pairwise magnitude
            pair_mag = pair_hidden.norm(dim=1).mean()   # scalar
            sum_mag  = sum_hidden.norm(dim=1).mean()
            head_mag = u.norm(dim=1).mean()
            self._last_aux = {
                "alloc": w.detach().cpu(),              # (B,M)
                "gate":  g.detach().cpu(),              # (B,M)
                "pair_mag": pair_mag.detach().cpu(),
                "sum_mag":  sum_mag.detach().cpu(),
                "head_mag": head_mag.detach().cpu(),
                "beta": float(self.beta.detach().cpu()),
            }

        return self.head(u)




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
