import torch
import torch.nn as nn

from models.utils import ChannelLayerNorm2d

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
        init_alpha: float = 0.0,
        dropout_p: float = 0.05,
        norm: bool = True,
    ):
        """
        :param num_channels: channels per stream after CNN backbone (C)
        :param reduction_ratio: SE reduction ratio
        :param init_alpha: initial value for the residual scaling alpha
        :param dropout_p: dropout probability on the residual path
        """
        super().__init__()
        self.P = num_person
        self.se = SEBlock(num_channels, reduction_ratio)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.dropout = nn.Dropout(p=dropout_p)
        self.pre_norm = ChannelLayerNorm2d(num_channels) if norm else nn.Identity()
        self.res_norm = ChannelLayerNorm2d(num_channels) if norm else nn.Identity()

        self._last_aux = None

    def get_aux(self):
        return self._last_aux

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

        # ---- pre-norm ----
        x_in = self.pre_norm(x).view(B, self.P, C, T, F)

        # ---- SE per person (masked) ----
        x_gated = self.se(
            x_in.view(B * self.P, C, T, F),
            None if mC is None else mC.view(B * self.P, C, T, F)
        ).view(B, self.P, C, T, F)

        # availability per person
        if mC is None:
            w = torch.ones(B, self.P, 1, T, F, device=x.device, dtype=x.dtype)
        else:
            w = (mC[:, :, :1] > 0).to(x.dtype)                  # (B,P,1,T,F)

        # others' masked mean (exclude self)
        xw = x_gated * w                                        # (B,P,C,T,F)
        sum_all = xw.sum(dim=1, keepdim=True)                   # (B,1,C,T,F)
        w_all   = w.sum(dim=1, keepdim=True).clamp_min(1e-6)    # (B,1,1,T,F)

        sum_others = sum_all - xw                                # (B,P,C,T,F)
        w_others   = (w_all - w).clamp_min(1e-6)                 # (B,P,1,T,F)
        others_mean = sum_others / w_others                      # (B,P,C,T,F)

        # ---- residual norm/dropout/scale and add ----
        res = self.res_norm(others_mean.view(B * self.P, C, T, F))
        res = self.dropout(res)
        res = res * self.alpha

        with torch.no_grad():
            avail = w.mean().detach().cpu()  # mean availability over all people
            res_mag = res.view(B, self.P, -1).norm(dim=2).mean().detach().cpu()
            self._last_aux = {
                "alpha": float(self.alpha.detach().cpu()),
                "avail_per_person": avail,
                "residual_mag": res_mag,
            }

        return x + res
