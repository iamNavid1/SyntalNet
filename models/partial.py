import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import Optional, Tuple

class PartialConv2d(nn.Module):
    """
    2D partial convolution for padded/missing data.
      - Convolves only over valid (mask==1) inputs.
      - Renormalizes output by kernel area / number of valid inputs.
      - Optionally propagates an updated mask of the same channel-dimension as input mask.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        return_mask: bool = False,
    ):
        super().__init__()

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in/out channels must be divisible by groups")
        
        # normalize to tuples
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.return_mask = return_mask
        self.kernel_area = float(kernel_size[0] * kernel_size[1])

        # base convolution (no bias in F.conv2d, we'll handle bias manually)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # buffer: ones-kernel to count valid inputs
        self.register_buffer("ones_kernel_shared", torch.ones(1, 1, *self.kernel_size))
        self.register_buffer("ones_kernel_group", torch.ones(groups, 1, *self.kernel_size))
        self.register_buffer("ones_kernel_per_channel", torch.ones(in_channels, 1, *self.kernel_size))

        # init
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            nn.init.zeros_(self.bias)

    def _expand(self, t: torch.Tensor, C_out: int) -> torch.Tensor:
        """Expand (B,C,H,W) to (B,C_out,H,W) by repeating each group of C evenly."""
        B, C, H, W = t.shape
        if C == 1 and C_out != 1:
            return t.expand(B, C_out, H, W)
        if C_out % C != 0:
            raise ValueError(f"Cannot expand {C}→{C_out}")
        mult = C_out // C
        return (
            t.unsqueeze(2)            # (B,C,1,H,W)
             .expand(-1, -1, mult, -1, -1)
             .reshape(B, C_out, H, W)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x:    (B, C_in, H, W)
        mask: (B, 1/C_in/G, H, W) binary (1=valid,0=pad) or None→all valid
        Returns:
          out: (B, C_out, H_out, W_out)
          updated_mask (if return_mask): (B, 1/G, H_out, W_out)
        """
        B, C, H, W = x.shape

        # prepare mask
        if mask is None:
            mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        else:
            mask = mask.to(device=x.device)
            if mask.dtype != x.dtype:
                mask = mask.to(dtype=x.dtype)
 
        C_mask = mask.shape[1]

        # determine mode
        if C_mask == 1:
            mode = "shared"
        elif C_mask == C and self.groups == 1:
            mode = "per_channel"
        elif C_mask == self.groups:
            mode = "per_group"
        # else:
        #     raise ValueError(
        #         f"Invalid mask channels={C_mask}. Expected 1, in_channels={C} (requires groups==1), or groups={self.groups}."
        #     )
        elif C_mask == C and self.groups > 1:
            # reduce to per-group
            B, _, H, W = mask.shape
            mask = mask.view(B, self.groups, C // self.groups, H, W).amax(dim=2)
            mode = "per_group"
        
        # multiply input by appropriate mask expansion
        if mode == "per_group":
            mask_x = self._expand(mask, C)   # (B,C,H,W)
        elif mode == "per_channel":
            mask_x = mask                                      # (B,C,H,W)
        else:  # shared
            mask_x = mask.expand(B, C, H, W)                   # (B,C,H,W)

        x_masked = x * mask_x

        # count valid inputs in each sliding window
        # update mask: 1 if any valid input in the window
        with torch.no_grad():
            if mode == "shared":
                wk = self.ones_kernel_shared.to(dtype=mask.dtype, device=mask.device)
                valid_count = F.conv2d(
                    mask, wk,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                )  # (B,1,H_out,W_out)
                updated_mask = (valid_count > 0).to(dtype=x.dtype)

            elif mode == "per_group":
                wk = self.ones_kernel_group[: self.groups].to(dtype=mask.dtype, device=mask.device)
                valid_count = F.conv2d(
                    mask, wk,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )  # (B,G,H_out,W_out)
                updated_mask = (valid_count > 0).to(dtype=x.dtype)

            else:  # per_channel (groups == 1)
                wk = self.ones_kernel_per_channel.to(dtype=mask.dtype, device=mask.device)
                valid_ch = F.conv2d(
                    mask, wk,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.in_channels,
                )  # (B,C_in,H_out,W_out)
                valid_count = valid_ch.sum(1, keepdim=True)  # (B,1,H_out,W_out)
                updated_mask = (valid_count > 0).to(dtype=x.dtype)  # mixed-channel updated mask

        # convolution
        if mode!='per_channel':
            # shared & per-group: fast conv+scale
            raw = F.conv2d(
                x_masked,
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            # spatial-only normalization: (B,1 or G, H_out, W_out)
            scale = torch.where(
                valid_count > 0,
                self.kernel_area / valid_count,
                torch.zeros_like(valid_count),
            )
            # apply renormalization
            if mode == "per_group":
                out = raw * self._expand(scale, self.out_channels)
            else:
                out = raw * scale  # broadcast over C_out
        else:
            # per-channel scales using unfold strategy
            kh,kw = self.kernel_size
            # unfold: (B, C*kh*kw, L)
            patches = F.unfold(
                x_masked, kernel_size=self.kernel_size,
                dilation=self.dilation, padding=self.padding,
                stride=self.stride
            )
            B,CK,L = patches.shape  # CK = C*kh*kw
            # compute per-channel scales (B,C,L)
            val_ch = valid_ch.reshape(B,C,-1)  # (B,C,L)
            scl_ch = torch.where(
                val_ch > 0, 
                self.kernel_area/val_ch,
                torch.zeros_like(val_ch),
            )
            # expand scl_ch to match patches
            scl_ch_ex = scl_ch.unsqueeze(2).expand(B, C, kh*kw, L).reshape(B, CK, L)
            patches = patches * scl_ch_ex
            # apply conv via matrix multiplication
            weight_flat = self.weight.reshape(self.out_channels, -1)  # (out, CKL)
            out = torch.matmul(weight_flat.unsqueeze(0), patches)  # (B, out, L)
            H_out = (H + 2*self.padding[0] - self.dilation[0]*(kh-1) -1)//self.stride[0] + 1
            W_out = (W + 2*self.padding[1] - self.dilation[1]*(kw-1) -1)//self.stride[1] + 1
            out = out.squeeze(0).reshape(B, self.out_channels, H_out, W_out)

        # bias handling 
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        # zero out fully-invalid locations
        gate = (valid_count > 0).to(dtype=x.dtype)
        if mode == "per_group":
            out = out * self._expand(gate, self.out_channels)
        else:
            out = out * gate

        if self.return_mask:
            return out, updated_mask
        return out

    def extra_repr(self):
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}, "
            f"return_mask={self.return_mask})"
        )


class PartialAvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.area = float(self.kernel_size[0] * self.kernel_size[1])

    def forward(self, x: torch.Tensor, m:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # m_in: (B,1 or >1,H,W)
        if m.shape[1] == 1:
            m_exp = m.expand(-1, x.shape[1], -1, -1)
        else:
            m_exp = m

        # sum of valid features in each window
        x_sum = F.avg_pool2d(x * m_exp, 
                            self.kernel_size, self.stride, self.padding,
                            ceil_mode=True) * self.area
        # number of valid entries per window
        m_cnt = F.avg_pool2d(m, 
                            self.kernel_size, self.stride, self.padding,
                            ceil_mode=True) * self.area

        # renormalize by valid count (avoid div by 0)
        x_out = torch.where(m_cnt > 0, x_sum / m_cnt.clamp_min(1.0), torch.zeros_like(x_sum))
        m_out = (m_cnt > 0).to(x.dtype)
        return x_out, m_out


class PartialGeM(nn.Module):
    """
    Generalized Mean pooling (global or local) with optional spatial mask over H*W (global 2D).
    
    Global mode (default):    (B,C,H,W) -> (B,C,1,1)
      - If mask `m` is given (B,1 or C,H,W), compute masked mean of x^p.

    Local mode (downsample):  (B,C,H,W) -> (B,C,H',W')
      - Uses windowed average of x^p, mask-aware via PartialAvgPool2d if mask is provided.
      - Falls back to nn.AvgPool2d when mask is None.
    """
    def __init__(
        self,
        kernel_size: int | Tuple[int, int] = None,
        stride: int | Tuple[int, int] = None,
        global_pool: bool = True,
        p: float = 3.0,
        eps: float = 1e-6,
        return_mask: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.global_pool = bool(global_pool)
        self.return_mask = bool(return_mask)

        # inverse-softplus init for q s.t. softplus(q) ≈ p
        q_init = torch.log(torch.expm1(torch.tensor(float(p))))
        self.q = nn.Parameter(q_init)

        if not self.global_pool:
            if kernel_size is None:
                raise ValueError("kernel_size must be set when global_pool=False")
            self.k = _pair(kernel_size)
            self.s = self.k if stride is None else _pair(stride)
            self._partial_pool = PartialAvgPool2d(self.k, self.s) if PartialAvgPool2d is not None else None

    def forward(self, x: torch.Tensor, m: Optional[torch.Tensor] = None):
        p = F.softplus(self.q) + self.eps  # ensure p > 0
        x_p = x.clamp(min=self.eps).pow(p)  # (B,C,H,W)

        # -------- GLOBAL GeM --------
        if self.global_pool:
            if m is None:
                pooled = F.adaptive_avg_pool2d(x_p, (1, 1))     # mean over HxW of x^p
            else:
                if m.size(1) == 1 and x_p.size(1) > 1:
                    m = m.expand(-1, x_p.size(1), -1, -1)
                m = m.to(dtype=x_p.dtype)

                num = (x_p * m).sum(dim=(2, 3), keepdim=True)
                den = m.sum(dim=(2, 3), keepdim=True)

                # identify empty masks per sample/channel
                empty = den <= self.eps
                den = den.clamp(min=self.eps)
                masked_mean = num / den

                pooled = torch.where(empty, torch.full_like(masked_mean, self.eps), masked_mean).clamp(min=self.eps)

            return pooled.pow(1.0 / p)

        # -------- LOCAL GeM --------
        if m is None:
            # standard avg pool on x^p
            pooled = F.avg_pool2d(x_p, self.k, self.s)
            out = pooled.clamp(min=self.eps).pow(1.0 / p)
            return out

        # mask-aware local pooling on x^p
        pooled, m_out = self._partial_pool(x_p, m)  # mean of x^p over valid positions
        pooled = pooled.clamp(min=self.eps).pow(1.0 / p)

        if self.return_mask:
            return pooled, m_out
        return pooled
