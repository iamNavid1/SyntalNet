import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from models.partial import PartialConv2d, PartialGeM, PartialMean2d
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
#                Social Squeeze-and-Excitation Cross-modal Fusion
# -----------------------------------------------------------------------------
class ChannelGate(nn.Module):
    """
    Channel-wise gating: squeezes spatial dimensions (T, F) to produce
    a per-channel gate, then expands back to (B, C, T, F).
    """
    def __init__(self, num_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(num_channels // reduction_ratio, 4)
        self.fc1 = nn.Linear(num_channels, hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, num_channels, bias=True)

    def forward(self, x: torch.Tensor, m: [torch.Tensor] = None) -> torch.Tensor:
        # x: (B,C,T,F); m: (B,1 or C,T,F) or None
        if m is not None:
            mC = (m if m.shape[1] == x.shape[1] else m.expand(-1, x.shape[1], -1, -1)).to(x.dtype)
            num = (x * mC).sum(dim=(2,3))
            den = mC.sum(dim=(2,3)).clamp_min(1e-6)
            y = num / den                         # (B,C) masked mean
        else:
            y = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)

        y = self.act(self.fc1(y))
        y = self.fc2(y)                           # (B,C)
        return y.view(x.shape[0], x.shape[1], 1, 1).expand_as(x)



class SpatialGate(nn.Module):
    """
    Spatial gating: squeezes channel dimension via 1x1 conv to produce
    a single-channel spatial mask, then expands to (B, C, T, F).
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, m: [torch.Tensor] = None) -> torch.Tensor:
        if m is not None:
            mC = (m if m.shape[1] == x.shape[1] else m.expand(-1, x.shape[1], -1, -1)).to(x.dtype)
            x_in = x * mC
        else:
            x_in = x
        y = self.conv(x_in)                       # (B,1,T,F)
        return y.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])


class SEBlock(nn.Module):
    """
    Concurrent channel-spatial Squeeze & Excitation Block with fusion
    """
    def __init__(
        self,
        num_channels: int,
        reduction_ratio: int = 16,
    ):
        super().__init__()
        self.channel_gate = ChannelGate(num_channels, reduction_ratio)
        self.spatial_gate = SpatialGate(num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, m: [torch.Tensor] = None) -> torch.Tensor:
        gc = self.channel_gate(x, m)              # (B,C,T,F)
        gs = self.spatial_gate(x, m)              # (B,C,T,F)
        return x * self.sigmoid(gc + gs)


class SoSE_X(nn.Module):
    """
    Social Squeeze-and-Excitation based Cross Fusion
    Applies a shared SE Block across P streams, then fuses each
    stream with the others via a learnable scalar residual coefficient.
    """
    def __init__(
        self,
        num_channels: int,
        reduction_ratio: int = 8,
        num_person: int = 3,
        init_temp: float = 0.0,
        num_alpha_groups: int = 8,
        alpha_max: float = 0.5,
        dropout_p: float = 0.05,
        p_drop_person: float = 0.15,
        norm: bool = True,
    ):
        """
        :param num_channels: channels per stream after CNN backbone (C)
        :param reduction_ratio: SE reduction ratio
        :param dropout_p: dropout probability on the residual path
        """
        super().__init__()
        self.P = num_person
        self.se = SEBlock(num_channels, reduction_ratio)
        self.dw = PartialConv2d(num_channels, num_channels, kernel_size=(3,1), padding=(1,0), groups=num_channels)
        self.dropout = nn.Dropout(p=dropout_p)
        self.p_drop_person = float(p_drop_person)
        self.pre_norm = ChannelLayerNorm2d(num_channels) if norm else nn.Identity()
        self.res_norm = ChannelLayerNorm2d(num_channels) if norm else nn.Identity()

        G = max(1, min(num_alpha_groups, num_channels))
        sizes = [(num_channels // G) + (1 if i < (num_channels % G) else 0) for i in range(G)]
        self.group_ids = torch.repeat_interleave(torch.arange(G), torch.tensor(sizes))
        self.alpha_raw = nn.Parameter(torch.zeros(G, dtype=torch.float32))
        self.alpha_max = float(alpha_max)
        self.temp_raw = nn.Parameter(torch.tensor(init_temp, dtype=torch.float32))

        self._last_aux = None

    def get_aux(self):
        return self._last_aux

    def _alpha_per_channel(self) -> torch.Tensor:
        temp = torch.nn.functional.softplus(self.temp_raw) + 1e-4
        alpha_g = self.alpha_max * torch.tanh(self.alpha_raw / temp)
        alpha_c = alpha_g[self.group_ids]
        return alpha_c

    def _maybe_person_dropout_mask(self, B: int, device, dtype) -> torch.Tensor:
        keep = torch.ones(B, self.P, 1, 1, 1, device=device, dtype=dtype)

        if not self.training or self.p_drop_person <= 0.0:
            return keep

        do_drop = (torch.rand(B, device=device) < self.p_drop_person)
        num_to_drop = int(do_drop.sum().item())
        if num_to_drop == 0:
            return keep

        drop_idx = torch.randint(low=0, high=self.P, size=(num_to_drop,), device=device)
        clip_ids = torch.nonzero(do_drop, as_tuple=False).view(-1)

        keep[clip_ids, drop_idx, :, :, :] = 0.0
        return keep

    def forward(self, x: torch.Tensor, m: [torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B*P, C, T, F)
        m: (B*P, 1 or C, T, F) or None
        """
        BP, C, T, F = x.shape
        assert BP % self.P == 0, "B*P mismatch in SoSE_X"
        B = BP // self.P

        # mask handling
        if m is not None:
            mB = m.view(B, self.P, m.shape[1], T, F)
            mC = (mB if mB.shape[2] == C else mB.expand(B, self.P, C, T, F)).to(x.dtype)
        else:
            mC = None

        # ---- person-dropout ----
        keep_mask = self._maybe_person_dropout_mask(B, x.device, x.dtype)

        # ---- pre-norm ----
        x_in = self.pre_norm(x)

        # ---- SE per person (masked) ----
        x_gated = self.se(
            x_in,
            None if mC is None else mC.view(B * self.P, C, T, F)
        ).view(B, self.P, C, T, F)

        # availability per person
        if mC is None:
            w = torch.ones(B, self.P, 1, T, F, device=x.device, dtype=x.dtype)
        else:
            w = (mC[:, :, :1] > 0).to(x.dtype)                  # (B,P,1,T,F)

        # apply person-dropout
        w = w * keep_mask

        # others' masked mean (exclude self)
        xw = x_gated * w                                        # (B,P,C,T,F)
        sum_all = xw.sum(dim=1, keepdim=True)                   # (B,1,C,T,F)
        w_all   = w.sum(dim=1, keepdim=True).clamp_min(1e-6)    # (B,1,1,T,F)

        sum_others = sum_all - xw                                # (B,P,C,T,F)
        w_others   = (w_all - w).clamp_min(1e-6)                 # (B,P,1,T,F)
        others_mean = sum_others / w_others                      # (B,P,C,T,F)
        others_mean = self.dw(
            others_mean.view(B*self.P, C, T, F),
            (w_others.view(BP, 1, T, F) > 0).to(x.dtype)
            # w.view(B*self.P, 1, T, F)
        )

        # ---- residual norm/dropout/scale and add ----
        res = self.res_norm(others_mean)
        # res = res / (res.pow(2).mean(dim=(1,2,3), keepdim=True).add_(1e-6)).sqrt()
        res = self.dropout(res)

        alpha = self._alpha_per_channel().view(1, C, 1, 1)
        res = res * alpha

        # bypass residual for dropped person
        res = res.view(B, self.P, C, T, F)
        res = res * keep_mask
        res = res.view(B * self.P, C, T, F)

        out = x_in + res
        # out = res

        # ---- aux diagnostics ----
        with torch.no_grad():
            avail = w.mean()  # mean availability over all people
            res_mag = res.view(B, self.P, -1).norm(dim=2).mean()
            x_mag = x_in.view(B, self.P, -1).norm(dim=2).mean()
            self._last_aux = {
                "alpha_mean": float(alpha.detach().cpu().mean()),
                "alpha_min": float(alpha.detach().cpu().min()),
                "alpha_max": float(alpha.detach().cpu().max()),
                "avail_per_person": float(avail.detach().cpu()),
                "residual_mag": float(res_mag.detach().cpu()),
                "x_mag": float(x_mag.detach().cpu()),
            }

        return out
