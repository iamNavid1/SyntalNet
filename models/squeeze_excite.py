import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import ChannelLayerNorm2d

class ChannelGate(nn.Module):
    """
    Channel-wise gating: squeezes spatial dimensions (T, F) to produce
    a per-channel gate, then expands back to (B, C, T, F).
    """
    def __init__(self, num_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden_channels = max(num_channels // reduction_ratio, 4)
        self.fc1 = nn.Linear(num_channels, hidden_channels, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, num_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        # squeeze spatial dims
        y = x.view(B, C, -1).mean(dim=2)              # (B, C)
        y = self.relu(self.fc1(y))                    # (B, hidden)
        y = self.fc2(y)                               # (B, C)
        # reshape and expand
        y = y.view(B, C, 1, 1).expand(B, C, T, F)     # (B, C, T, F)
        return y


class SpatialGate(nn.Module):
    """
    Spatial gating: squeezes channel dimension via 1x1 conv to produce
    a single-channel spatial mask, then expands to (B, C, T, F).
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        y = self.conv(x)                              # (B, 1, T, F)
        y = y.expand(B, C, T, F)                      # (B, C, T, F)
        return y


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F)
        g_c = self.channel_gate(x)      # (B, C, T, F)
        g_s = self.spatial_gate(x)      # (B, C, T, F)
        gate = self.sigmoid(g_c + g_s)  # (B, C, T, F)
        return x * gate


class CrossSE(nn.Module):
    """
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
        norm: bool = False,
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
        # learnable scalar controlling how much of the others' features to add
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm = ChannelLayerNorm2d(num_channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor of shape (BP, C, T, F)
        :return: tensor of same shape with cross-person fusion applied
        """
        BP, C, T, F = x.shape
        P = self.P
        B = BP // P

        # 1) Shared gating: apply SE block to each stream
        x_gated = self.se(x)                  # (B⋅P, C, T, F)
        x_gated = x_gated.view(B, P, C, T, F)   # (B, P, C, T, F)

        # 2) Cross-person residual aggregation
        #    for each p, sum the gated features of all others
        sum_all = x_gated.sum(dim=1, keepdim=True) # (B, 1, C, T, F)
        # subtract self to get sum_{j≠p}
        others_sum = sum_all - x_gated             # (B, P, C, T, F)
        others_mean = others_sum / max(1, P-1)

        # 3) norm, dropout, scale, residual
        out = self.norm(others_mean.view(BP, C, T, F))
        out = self.dropout(out)
        out = self.alpha * out
        out = x + out

        return out


# === Example usage ===

if __name__ == "__main__":
    Bsz       = 8
    num_people = 3
    C_in      = 1        # raw input channels
    seq_len   = 190      # temporal dimension
    dim_feat  = 1024     # feature dimension

    # 1) A simple CNN backbone per person
    class SimpleBackbone(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            # x: (B, C_in, T, F)
            return self.conv(x)  # (B, out_channels, T, F)

    # 2) Full multi-person model
    class MultiPersonModel(nn.Module):
        def __init__(self, C_backbone=64):
            super().__init__()
            self.backbone = SimpleBackbone(C_in, C_backbone)
            self.fusion   = CrossSE(
                num_channels=C_backbone,
                reduction_ratio=16,
                init_alpha=0.0,
                dropout_p=0.05
            )
            # example task head: global pooling + classifier
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),    # pool (T, F) to (1,1)
                nn.Flatten(),                    # (B⋅P, C_backbone)
                nn.Linear(C_backbone, 10)        # e.g., 10 classes per person
            )

        def forward(self, x):
            # x: (B, P, C_in, T, F)
            B, P, C_in, T, F = x.shape
            # apply backbone per person by folding P into batch
            x_flat = x.view(B * P, C_in, T, F)
            feat    = self.backbone(x_flat)           # (B⋅P, C_backbone, T, F)
            feat    = feat.view(B, P, feat.size(1), T, F)

            # shared CSSE + cross-person fusion
            fused = self.fusion(feat)                 # (B, P, C_backbone, T, F)

            # classification head per person
            fused_flat = fused.view(B * P, feat.size(1), T, F)
            logits     = self.head(fused_flat)        # (B⋅P, 10)
            logits     = logits.view(B, P, -1)        # (B, P, 10)
            return logits

    # instantiate and test
    model = MultiPersonModel(num_people)
    dummy = torch.randn(Bsz, num_people, C_in, seq_len, dim_feat)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (Bsz, num_people, 10)
