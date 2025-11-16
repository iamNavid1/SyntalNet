import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from models.partial import PartialConv2d
from models.utils   import ChannelLayerNorm2d


def _expand_mask(m: torch.Tensor, C: int, dtype) -> torch.Tensor:
    if m.dim() == 5:
        B, P, c, t, f = m.shape
        m_reshaped = m.reshape(B * P, c, t, f)
        out = m_reshaped if c == C else m_reshaped.expand(-1, C, -1, -1).to(dtype)
        return out.reshape(B, P, C, t, f)
    return m if m.shape[1] == C else m.expand(-1, C, -1, -1).to(dtype)

class GRN(nn.Module):
    """
    Mask-aware GRN (Global Response Normalization) layer
    """
    def __init__(self, C: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.eps   = eps


    def forward(self, x: torch.Tensor, m: Optional[torch.Tensor] = None) -> torch.Tensor:
        if m is None:
            Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        else:
            mC = _expand_mask(m, x.shape[1], x.dtype)
            Gx = torch.sqrt(((x * x) * mC).sum(dim=(2, 3), keepdim=True) + self.eps)
            valid_cnt = mC.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            Gx = Gx * (valid_cnt > 0).to(Gx.dtype)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


class FMixLowRank(nn.Module):
    """
    Low-rank feature mixer
    """
    def __init__(
        self,
        F_in: int,
        F_out: Optional[int] = None,
        rank: Optional[int] = None,
        alpha: float = 1.e-2,
        act: nn.Module = nn.Identity,
        p_drop: float = .05,
        residual: bool = True,
        pre_norm: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.F_in = F_in
        self.F_out = F_out or self.F_in
        self.rank = rank or max(4, min(16, self.F_out // 8))
        self.maybe_act = act()
        self.drop = nn.Dropout(p_drop)
        self.residual = bool(residual and (self.F_in == self.F_out))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32)) if residual else None
        self.pre_norm = pre_norm
        self.bias = bias

        if self.F_out != self.F_in:
            self.pre_norm = True
            self.bias = True

        self.maybe_pre_norm = nn.LayerNorm(self.F_in) if self.pre_norm else nn.Identity()
        self.U = nn.Linear(self.F_in, self.rank, bias=False)
        self.V = nn.Linear(self.rank, self.F_out, bias=self.bias)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"Expected x of shape (B,C,T,F), got {tuple(x.shape)}"
        B, C, T, Fsize = x.shape
        assert Fsize == self.F_in, f"Last dim {Fsize} != F_in {self.F_in}"

        X = x.reshape(B * C * T, self.F_in)     # (N, F_in)
        X = self.maybe_pre_norm(X)              # (N, F_in)
        H = self.U(X)                           # (N, r)
        H = self.maybe_act(H)                   # (N, r)
        H = self.drop(H)                        # (N, r)
        Y = self.V(H)                           # (N, F_out)

        if self.residual:
            Y = F.softplus(self.alpha) * Y.view(B,C,T,self.F_out)
            out = x + Y
        else:
            out = Y.reshape(B, C, T, self.F_out)

        if m is not None:
            any_valid = (m > 0).any(dim=-1, keepdim=True)
            m = any_valid.to(dtype=m.dtype).expand(B, C, T, self.F_out)

        return out, m


class StageTransition(nn.Module):
    """
    Mask-aware 1x1 projection
    """
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True, norm: bool = False, p_drop: float = 0.0):
        super().__init__()
        self.norm = ChannelLayerNorm2d(in_ch) if norm else nn.Identity()
        self.proj = PartialConv2d(in_ch, out_ch, kernel_size=1, bias=bias, return_mask=False)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.proj(self.norm(x), m)
        out = self.drop(out)
        return out, m


class CNXv2Block(nn.Module):
    """
    Pre-norm ConvNeXt-v2-style block (mask-aware):
    """
    def __init__(
        self,
        C: int,
        k: int,
        p: int,
        s: int,
        d: int = 1,
        expansion: int | float = 1.5,
        act: nn.Module = nn.GELU,
        p_drop: float = 0.05,
        p_droppath: float = 0.0,
    ):
        super().__init__()
        self.prenorm = ChannelLayerNorm2d(C)
        self.dw = PartialConv2d(
            in_channels=C, out_channels=C,
            kernel_size=k, stride=s, padding=p, dilation=d,
            groups=C, return_mask=False, bias=False,
        )
        hidden = int(C * expansion)
        self.pw1 = nn.Conv2d(C, hidden, kernel_size=1, bias=True)
        self.act = act()
        # self.grn = GRN(hidden)
        self.drop = nn.Dropout(p_drop)
        self.pw2 = nn.Conv2d(hidden, C, kernel_size=1, bias=True)
        self.drop_path = DropPath(p_droppath) if p_droppath > 0.0 else nn.Identity()
        nn.init.zeros_(self.pw2.weight)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = x
        y = self.prenorm(x)
        y = self.dw(y, m)
        y = self.pw1(y)
        y = self.act(y)
        # y = self.grn(y, m)
        y = self.drop(y)
        y = self.pw2(y)
        y = self.drop_path(y)
        out = identity + y
        out = out * m
        return out, m
