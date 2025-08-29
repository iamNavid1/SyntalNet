import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple
from typing import Tuple, Sequence, Union


class MLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            p_drop: float = 0.2,
            act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or max(64, in_features // 2)

        self.ln = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x


class ChannelLayerNorm2d(nn.Module):
    """
    LayerNorm over channels for 4D tensors (B, C, T, F).
    """
    def __init__(self, C: int):
        super().__init__()
        self.ln = nn.LayerNorm(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()       # (B, T, F, C)
        x = self.ln(x)                               # LN over channels only
        return x.permute(0, 3, 1, 2).contiguous()    # back to (B, C, T, F)


def _normalize_out_channels(v: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Return a 3-tuple of ints."""
    trip = _triple(v)
    if not all(isinstance(x, int) for x in trip):
        raise ValueError("out_channels must resolve to a 3-tuple of ints.")
    return trip  # type: ignore[return-value]


def _normalize_in_channels(
    v: Union[int, Tuple[int, int, int]],
    out_channels: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    If a single int is provided, assume stage1=in, stage2=out1, stage3=out2
    (a common pattern for sequential conv stacks).
    """
    if isinstance(v, int):
        return (v, out_channels[0], out_channels[1])
    trip = _triple(v)
    if not all(isinstance(x, int) for x in trip):
        raise ValueError("in_channels must be an int or a 3-tuple of ints.")
    return trip  # type: ignore[return-value]


def _normalize_spatial(v, name):
    """
    Normalize:
        - int -> ((i,i),(i,i),(i,i))
        - (h,w) -> ((h,w),(h,w),(h,w))
        - ((h1,w1),(h2,w2),(h3,w3)) -> as-is (validated)
    """
    # case: int or (h, w)
    if isinstance(v, int) or (isinstance(v, Sequence) and len(v) == 2 and all(isinstance(x, int) for x in v)):  # type: ignore[arg-type]
        p = _pair(v)  # (h, w)
        return (p, p, p)

    # case: ((h1,w1),(h2,w2),(h3,w3))
    if isinstance(v, Sequence) and len(v) == 3:
        vv = tuple(_pair(x) for x in v)  # validates each sub-pair
        # ensure each element is a 2-int tuple
        if all(isinstance(a, int) and isinstance(b, int) for (a, b) in vv for _ in (0,)):
            return vv  # type: ignore[return-value]

    raise ValueError(
        f"{name} must be an int, a 2-tuple, or a 3-tuple of 2-tuples; got {v!r}"
    )


def _preserves_spatial(
          ks: int | Tuple[int, int],
          st: int | Tuple[int, int],
          pd: int | Tuple[int, int],
          dl: int | Tuple[int, int] = 1
) -> tuple[bool, bool, bool]:
    ks = _pair(ks)
    st = _pair(st)
    pd = _pair(pd)
    dl = _pair(dl)
    ph = (st[0] == 1) and (2 * pd[0] == dl[0] * (ks[0] - 1))
    pw = (st[1] == 1) and (2 * pd[1] == dl[1] * (ks[1] - 1))
    return ph and pw


def _is_same_like(
          ks: int | Tuple[int, int],
          st: int | Tuple[int, int],
          pd: int | Tuple[int, int],
          dl: int | Tuple[int, int] = 1
) -> bool:
    ks = _pair(ks)
    st = _pair(st)
    pd = _pair(pd)
    dl = _pair(dl)
    return (st[0] == 1 or 2*pd[0] == dl[0]*(ks[0]-1)) and \
           (st[1] == 1 or 2*pd[1] == dl[1]*(ks[1]-1))
