import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.modules.utils import _pair
from typing import List, Tuple, Sequence, Union, Optional, Dict

import models.utils as mutils
from models.base_model import BaseModel
from models.squeeze_excite import CrossSE
from models.partial import PartialConv2d, PartialAvgPool2d
from models.fusion import GPSFusion, GLRFusion


class CNNBackbone(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 32,
            kernel_size: int | Tuple[int, int] = 5,
            stride: int | Tuple[int, int] = 1,
            padding: int | Tuple[int, int] = 2,
            dilation: int | Tuple[int, int] = 1,
            p_drop: float = 0.05,
            activation: str = "silu",
            cross_se: bool = True,
            residual: bool = True,
            mask_aware_skip: bool = True,
    ):
        super().__init__()

        self.prenorm = mutils.ChannelLayerNorm2d(in_channels)
        self.act0 = nn.SiLU(inplace=True) if activation == "silu" else nn.GELU()
        self.conv = PartialConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            return_mask  = True,
            )
        self.drop = nn.Dropout(p_drop)
        self.skip = None
        self.mp_fusion = None  # multi-person channel fusion

        if cross_se:
            self.mp_fusion = CrossSE(
                num_channels    = out_channels,
                reduction_ratio = 8, 
            )

        self.mask_aware_skip = mask_aware_skip
        self._skip_mode, self.skip = self._make_residual(
            residual=residual,
            mask_aware_skip=mask_aware_skip,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @staticmethod
    def _make_residual(*, residual: bool, mask_aware_skip: bool, in_channels: int, out_channels: int,
        kernel_size: int | Tuple[int, int], stride: int | Tuple[int, int],
        padding: int | Tuple[int, int], dilation: int | Tuple[int, int],
    ) -> Tuple[str, Optional[nn.Module]]:
        
        if not residual:
            return None, None

        stride = _pair(stride)
        preserves = mutils._preserves_spatial(kernel_size, stride, padding, dilation)
        need_mask_update = (stride != (1, 1))
        ceil = mutils._is_same_like(kernel_size, stride, padding, dilation)

        if mask_aware_skip:
            # ----- mask-aware skip -----
            if not preserves:
                # downsample + 1x1 projection (mask-aware)
                skip = nn.ModuleList([
                    PartialAvgPool2d(stride, stride),
                    PartialConv2d(in_channels, out_channels, kernel_size=1,
                                  bias=False, return_mask=need_mask_update),
                ])
                return "partial_pool_1x1", skip

            if in_channels != out_channels:
                # 1x1 projection (mask-aware)
                skip = PartialConv2d(in_channels, out_channels, kernel_size=1,
                                     bias=False, return_mask=need_mask_update)
                return "partial_1x1", skip

            # identity (mask-aware path in forward)
            return "partial_identity", None

        # ----- vanilla skip -----
        if not preserves:
            skip = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=ceil),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
            return "vanilla_pool_1x1", skip

        if in_channels != out_channels:
            skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            return "vanilla_1x1", skip

        return "vanilla_identity", nn.Identity()

    def _residual_forward(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._skip_mode in ["vanilla_pool_1x1", "vanilla_1x1", "vanilla_identity"]:
            return self.skip(x), m

        if self._skip_mode == "partial_identity":
            m_exp = self._expand_to_channels(m, x.shape[1], x.dtype)
            return x * m_exp, m

        if self._skip_mode == "partial_pool_1x1":
            x, m = self.skip[0](x, m)
            out = self.skip[1](x, m)
            return out if isinstance(out, tuple) else (out, m)
        
        # partial_1x1 skip
        out = self.skip(x, m)
        return out if isinstance(out, tuple) else (out, m)

    @staticmethod
    def _to_shared_mask(m: torch.Tensor, dtype) -> torch.Tensor:
        return m if m.shape[1] == 1 else (m.sum(dim=1, keepdim=True) > 0).to(dtype)

    @staticmethod
    def _expand_to_channels(m: torch.Tensor, C: int, dtype) -> torch.Tensor:
        return m if m.shape[1] == C else m.expand(-1, C, -1, -1).to(dtype)

    def forward(
            self, 
            x: torch.Tensor,  # (B, P, C, T, F) 
            m: torch.Tensor   # (B, P, 1, T, F)
    ) -> torch.Tensor:
        
        # folding P into batch
        B, P, C, T, F = x.shape
        x = x.view(B * P, C, T, F)
        m = m.view(B * P, 1, T, F)

        # apply convolution pipeline
        out = self.act0(self.prenorm(x))
        out, mask = self.conv(out, m)
        out = self.drop(out)

        # apply multi-person channel fusion
        if self.mp_fusion is not None:
            # fusion expects (B,P,C',T',F')
            out = out.view(B, P, out.shape[1], out.shape[2], out.shape[3])
            out = self.mp_fusion(out)
            out = out.view(B * P, out.shape[2], out.shape[3], out.shape[4])

        # apply skip connection
        if self.skip is not None:
            s, m_s = self._residual_forward(x, m)  # (B*P, C', T', F')
            if self.mask_aware_skip:
                out += s
                m_s_shared = self._to_shared_mask(m_s, out.dtype)
                mask = torch.maximum(m_s_shared, mask)
            else:
                s *= mask  # mask-gated residual unit
                out += s

       # back to (B, P, C', T', F')
        out = out.view(B, P, out.shape[-3], out.shape[-2], out.shape[-1])
        mask = mask.view(B, P, 1, out.shape[-2], out.shape[-1])

        return out, mask


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


class Branch(nn.Module):
    def __init__(
            self,
            in_ch_emb: int,
            in_ch_feat: int,
            dim_emb: int,
            dim_feat: int,
            num_blocks: int = 3,
            upsample_emb: bool = False,
            shared_dim: int = 128,
            shared_out_ch: int = 32,
            ch_expansion: List[int] = [2, 2],
            ks_emb: List[int | Tuple[int, int]] = [5, 5, 5],
            ks_feat: List[int | Tuple[int, int]] = [5, 5, 5],
            st_emb: List[int | Tuple[int, int]] = [1, 1, 1],
            st_feat: List[int | Tuple[int, int]] = [1, 1, 1],
            pd_emb: List[int | Tuple[int, int]] = [2, 2, 2],
            pd_feat: List[int | Tuple[int, int]] = [2, 2, 2],
            dl_emb: List[int | Tuple[int, int]] = [1, 1, 1],
            dl_feat: List[int | Tuple[int, int]] = [1, 1, 1],
            p_drop_block: float = 0.05,
            activation_block: str = nn.SiLU,
            residual: bool = True,
            cross_se: bool = True,
            p_drop_proj: float = 0.05,
            activation_proj: str = nn.GELU,
    ):
        super().__init__()
        assert all(num_blocks == len(lst) for lst in [ks_emb, ks_feat, st_emb, st_feat, pd_emb, pd_feat, dl_emb, dl_feat]), \
            f"Expected length {num_blocks} for CNN backbone inputs"
        
        if upsample_emb:
            self.emb_upsampler = nn.ConvTranspose1d(
                in_channels = dim_emb,
                out_channels = dim_emb,
                kernel_size = 23,
                stride = 3,
                padding = 1,
                output_padding = 1,
                groups=dim_emb,
                bias=False,
            )
        else:
            self.emb_upsampler = None

        self.blocks_emb = nn.ModuleList([
            CNNBackbone(
                in_ch_emb if i == 0 else shared_out_ch, shared_out_ch,
                ks_emb[i], st_emb[i], pd_emb[i], dl_emb[i],
                p_drop_block, activation_block, 
                residual, cross_se,
            ) for i in range(num_blocks)
        ])

        self.blocks_feat = nn.ModuleList([
            CNNBackbone(
                in_ch_feat if i == 0 else shared_out_ch, shared_out_ch,
                ks_feat[i], st_feat[i], pd_feat[i], dl_feat[i],
                p_drop_block, activation_block, 
                residual, cross_se,
            ) for i in range(num_blocks)
        ])

        self.feat_projector = MLP(
            in_features=dim_feat, hidden_features=32, out_features=shared_dim, 
            p_drop=p_drop_proj,act_layer=activation_proj
        )

        self.mc_fusion = GPSFusion(C_mod=shared_out_ch, C_exp=ch_expansion)  # intramodality multi-channel fusion

    def cnn_block_forward(self, x, m, blocks):
        out = x
        for block in blocks:
            out, m = block(out, m)
        return out, m

    def forward(
            self,
            x_feat: List[torch.Tensor],  # (B, P, T, F_f_i)
            m_feat: List[torch.Tensor],  # (B, P, T, F_f_i)
            x_emb: torch.Tensor,         # (B, P, (T)(T_e), F_e)
            m_emb: torch.Tensor,         # (B, P, (T)(T_e), F_e)
    ) -> torch.Tensor:
        
        B, P, _, _ = x_emb.shape

        x_feat = torch.cat(x_feat, dim=-1).unsqueeze(2)
        m_feat = torch.cat(m_feat, dim=-1).unsqueeze(2)

        z_feat, m_feat = self.cnn_block_forward(x_feat, m_feat, self.blocks_feat)
        z_feat = z_feat.view(B * P, z_feat.shape[-3], z_feat.shape[-2], z_feat.shape[-1])
        z_feat = self.feat_projector(z_feat)
        m_feat = m_feat.view(B * P, m_feat.shape[-3], m_feat.shape[-2], m_feat.shape[-1])
        m_feat = (m_feat.sum(dim=-1, keepdim=True) > 0).expand(B * P, m_feat.shape[-3], m_feat.shape[-2], z_feat.shape[-1])
        # z_feat = z_feat.view(B, P, z_feat.shape[-3], z_feat.shape[-2], z_feat.shape[-1])

        if self.emb_upsampler:
            x_emb = x_emb.view(B * P, x_emb.shape[-2], x_emb.shape[-1]).transpose(1, 2)
            x_emb = self.emb_upsampler(x_emb)
            x_emb = x_emb.view(B, P, x_emb.shape[-2], x_emb.shape[-1]).transpose(2, 3)
        x_emb = x_emb.unsqueeze(2)
        m_emb = m_emb.unsqueeze(2)

        z_emb, m_emb = self.cnn_block_forward(x_emb, m_emb, self.blocks_emb)
        z_emb = z_emb.view(B * P, z_emb.shape[-3], z_emb.shape[-2], z_emb.shape[-1])
        m_emb = m_emb.view(B * P, m_emb.shape[-3], m_emb.shape[-2], m_emb.shape[-1])
        
        z_branch = self.mc_fusion(z_feat, z_emb, m_feat, m_emb)

        return z_branch


class CosineProtoClassifier(nn.Module):
    """
    Per-label K-class cosine classifier with optional EMA prototypes and per-class fusion.
    Safe features:
      - temperature scaling
      - warm-up for prototype fusion
      - mask unseen prototypes early
    """
    def __init__(
            self, 
            dim: int, 
            num_classes: int = 3,
            init_scale_param: float = 16.0,
            use_prototypes: bool = True,
            init_scale_proto: float = 10.0,
            proto_momentum: float = 0.99,
            fuse_init: float = -1.5,
            warmup_epochs: int = 5,
            learn_temperature: bool = True,
            init_temperature: float = 1.0,
    ):
        super().__init__()
        self.D = dim
        self.K = num_classes

        # parametric cosine classifier
        self.W = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.normal_(self.W, std=0.02)
        self.log_scale_param = nn.Parameter(torch.log(torch.tensor(init_scale_param)))

        # prototype cosine classifier
        self.use_prototypes = use_prototypes
        self.m = proto_momentum
        self.warmup_epochs = warmup_epochs
        if use_prototypes:
            self.register_buffer("prototypes", F.normalize(torch.randn(num_classes, dim), dim=1))
            self.register_buffer("proto_counts", torch.zeros(num_classes))
            self.log_scale_proto = nn.Parameter(torch.log(torch.tensor(init_scale_proto)))
            self.fuse_logit = nn.Parameter(torch.full((num_classes,), fuse_init))  # per-class λ in (0,1)

        # temperature
        self.learn_temperature = learn_temperature
        if learn_temperature:
            self.log_T = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        else:
            self.register_buffer("T_buffer", torch.tensor(init_temperature))
            self.log_T = None

    @staticmethod
    def _cos_logits(z: torch.Tensor, W: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        W = F.normalize(W, dim=-1)
        return scale * (z @ W.t())  # (B,K)

    @torch.no_grad()
    def update_prototypes(self, z: torch.Tensor, y: torch.Tensor):
        if not self.use_prototypes:
            return
        
        z = F.normalize(z.float(), dim=-1)
        y = y.long()
        K, D = self.K, self.D

        # local class sums & counts
        mask = F.one_hot(y, num_classes=K).to(z.dtype)
        sums = mask.t().matmul(z)
        cnts = mask.sum(dim=0)

        # all-reduce to GLOBAL sums/cnts
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnts, op=dist.ReduceOp.SUM)

        # EMA update
        nz = (cnts > 0)
        if nz.any():
            means = sums[nz] / cnts[nz].unsqueeze(1).clamp_min(1e-6)
            proto = self.prototypes[nz]
            updated = self.m * proto + (1.0 - self.m) * means
            self.prototypes[nz] = F.normalize(updated, dim=1).to(self.prototypes.dtype)
        self.proto_counts += cnts.to(self.proto_counts.dtype)

    def forward(self, z: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        s_param = torch.exp(self.log_scale_param).clamp(1., 100.)
        logits_param = self._cos_logits(z, self.W, s_param)

        if self.use_prototypes:
            s_proto = torch.exp(self.log_scale_proto).clamp(1., 100.)
            logits_proto = self._cos_logits(z, self.prototypes, s_proto)
            # mask unseen classes for prototypes
            seen = (self.proto_counts > 0).float()
            if (seen == 0).any():
                logits_proto = logits_proto * seen.unsqueeze(0) + logits_param * (1 - seen).unsqueeze(0)

            lam = torch.sigmoid(self.fuse_logit)  # (K,)
            if epoch is not None and epoch < self.warmup_epochs:
                lam = lam * 0.0
            logits = (1 - lam.unsqueeze(0)) * logits_param + lam.unsqueeze(0) * logits_proto
        else:
            logits = logits_param

        # add temperature
        T = (torch.exp(self.log_T) if self.log_T is not None else self.T_buffer).clamp(1e-3, 100.0)
        return logits / T


class ClassificationHead(nn.Module):
    """
    Shared trunk + per-label adapters + per-label cosine/proto classifiers.
    """
    def __init__(
            self,
            dim: int,
            head_names: List[str],
            num_classes: int = 3,
            trunk_hidden: Optional[int] = None, 
            p_drop_trunk: float = 0.2,
            adapter_hidden: Optional[int] = None, 
            p_drop_adapter: float = 0.1,
            residual: bool = True,
            use_prototypes: bool = True, 
            proto_momentum: float = 0.99, 
            warmup_epochs: int = 5,
            group_wise: bool = False,
            person_per_grp: int = 3,
    ):
        super().__init__()
        self.heads = head_names
        self.P = person_per_grp
        
        self.group_wise = group_wise
        if group_wise:
            dim = dim * person_per_grp
        
        self.residual = residual
        if residual:
            self.alpha = nn.Parameter(torch.full((dim,), 1e-4))
        
        self.trunk = MLP(dim, trunk_hidden, p_drop=p_drop_trunk)

        self.adapters = nn.ModuleDict({name: MLP(dim, adapter_hidden, p_drop=p_drop_adapter)
                                       for name in self.heads})
        self.classifiers = nn.ModuleDict({name: CosineProtoClassifier(dim, num_classes=num_classes,
                                                                      use_prototypes=use_prototypes,
                                                                      proto_momentum=proto_momentum,
                                                                      warmup_epochs=warmup_epochs)
                                          for name in self.heads})

    @torch.no_grad()
    def update_prototypes(self, z_dict: Dict[str, torch.Tensor], y_dict: Dict[str, torch.Tensor]):
        for name in self.heads:
            if self.group_wise:
                B, D = z_dict[name].shape
                B = B // self.P
                z_dict[name] = z_dict[name].view(B, self.P, D).reshape(B, self.P * D)
            self.classifiers[name].update_prototypes(z_dict[name], y_dict[name])

    def forward(self, z: torch.Tensor, epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if self.group_wise:
            B, D = z.shape
            B = B // self.P
            z = z.view(B, self.P, D).reshape(B, self.P * D)
        
        z_shared = self.trunk(z)
        
        if self.residual:
            z_shared = z + self.alpha * z_shared 

        z_head: Dict[str, torch.Tensor] = {}
        logits: Dict[str, torch.Tensor] = {}
        for name in self.heads:
            z_out = self.adapters[name](z_shared)
            if self.residual:
                z_out = z_shared + self.alpha * z_out
            z_head[name] = z_out
            logits[name] = self.classifiers[name](z_out, epoch=epoch)

        features: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {
            'shared': z_shared,
            'per_head': z_head,
        }
        return features, logits


class SyntalNet(BaseModel):
    def __init__(
            self,
            branches: List[str]         = ['Videokinetic', 'Dialogue', 'Acoustic'],
            dim_out: int                = 256,
            in_ch_emb: List[int]        = [1, 1, 1],
            in_ch_feat: List[int]       = [1, 1, 1],
            dim_emb: List[int]          = [1024, 1024, 512],
            dim_feat: List[int]         = [394, 12, 12],
            num_blocks: List[int]       = [3, 3, 3],
            upsample_emb: bool          = [True, False, False],
            shared_dim: int             = 128,
            shared_out_ch: int          = 32,
            ch_expansion: List[int]     = [2, 2],
            stride_emb: List[List[int]] = [[2, 2, 2], [2, 2, 2], [1, 2, 2]],
            branch_residual: bool       = True,
            cross_se: bool              = True,
            ind_cls_heads: Tuple[str]   = (),
            grp_cls_heads: Tuple[str]   = (),
            use_prototypes: bool        = True,
            proto_warmup_epochs: int    = 5,
            cls_residual: bool          = True,
    ):
        super().__init__()

        assert all(branch in ['Videokinetic', 'Dialogue', 'Acoustic'] for branch in branches), \
            "Branches must be one of 'Videokinetic', 'Dialogue', or 'Acoustic'"
        assert len(branches) == len(in_ch_emb) == len(in_ch_feat) == len(dim_emb) == len(dim_feat) == len(num_blocks) == len(upsample_emb), \
            "All input lists (branches, in_ch_emb, in_ch_feat, dim_emb, dim_feat, num_blocks, upsample_emb) must have the same length"

        self.branches = nn.ModuleDict()
        for i, branch in enumerate(branches):
            self.branches[branch] = Branch(
                in_ch_emb     = in_ch_emb[i],
                in_ch_feat    = in_ch_feat[i],
                dim_emb       = dim_emb[i],
                dim_feat      = dim_feat[i],
                num_blocks    = num_blocks[i],
                upsample_emb  = upsample_emb[i],
                shared_dim    = shared_dim,
                shared_out_ch = shared_out_ch,
                st_emb        = [(1, s) for s in stride_emb[i]],
                residual      = branch_residual,
                cross_se      = cross_se,
            )

        self.mm_fusion = GLRFusion(
            dims_mod     = [2 * shared_out_ch * ch_expansion[0] * ch_expansion[1] for _ in range(len(branches))],
            dim_hidden   = shared_out_ch * ch_expansion[0] * ch_expansion[1],
            rank_pair    = shared_out_ch,
            alloc_hidden = 2 * shared_out_ch,
            dim_out      = dim_out,
        )

        self.individual_classifier = None
        self.group_classifier = None

        if ind_cls_heads:
            self.individual_classifier = ClassificationHead(
                dim            = dim_out,
                head_names     = list(ind_cls_heads),
                use_prototypes = use_prototypes,
                residual       = cls_residual,
                warmup_epochs  = proto_warmup_epochs,
            )

        if grp_cls_heads:
            self.group_classifier = ClassificationHead(
                dim            = dim_out,
                head_names     = list(grp_cls_heads),
                trunk_hidden   = dim_out,
                adapter_hidden = dim_out,
                use_prototypes = use_prototypes,
                residual       = cls_residual,
                warmup_epochs  = proto_warmup_epochs,
                group_wise     = True,
            )

    @torch.no_grad()
    def update_prototypes(
        self,
        z_dict: Dict[torch.Tensor | str, Dict[str, torch.Tensor]], 
        y_dict: Dict[str, Dict[str, torch.Tensor]],
    ):
        if self.individual_classifier is not None:
            z = z_dict["individual"]["per_head"]
            y = y_dict["individual"]
            assert y, "No individual logits found in y_dict['individual']"
            self.individual_classifier.update_prototypes(z, y)
        if self.group_classifier is not None:
            z = z_dict["group"]["per_head"]
            y = y_dict["group"]
            assert y, "No group logits found in y_dict['group']"
            self.group_classifier.update_prototypes(z, y)

    def forward(self, batch_data, epoch = None):

        z_branch = []
        for branch_name, branch_module in self.branches.items():
            x_feat, m_feat, x_emb, m_emb = batch_data[branch_name.lower()]
            z_branch.append(branch_module(x_feat, m_feat, x_emb, m_emb))

        z = self.mm_fusion(z_branch)

        z_outs: Dict[torch.Tensor | Dict[str, torch.Tensor]] = {}
        logits: Dict[str, Dict[str, torch.Tensor]] = {}
        z_outs['backbone'] = z
        if self.individual_classifier:
            ind_features, ind_logits = self.individual_classifier(z, epoch)
            z_outs.update({'individual': ind_features})
            logits.update({'individual': ind_logits})
        if self.group_classifier:
            grp_features, grp_logits = self.group_classifier(z, epoch)
            z_outs.update({'group': grp_features})
            logits.update({'group': grp_logits})

        return z_outs, logits







# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple

# class RobustBinaryHead(nn.Module):
#     """
#     Robust binary classification head for fused embeddings.
#     Input:  x ∈ R^{B×D_in}  (e.g., output of GLRFusion)
#     Output: logits ∈ R^{B×1}

#     Features:
#       - Pre-norm + small residual MLP (SiLU)
#       - Cosine classifier (norm-invariant) with learnable scale s
#       - Optional temperature scaling (learned or set post-hoc)
#       - Optional logit-prior adjustment for class imbalance
#     """
#     def __init__(
#         self,
#         dim_in: int,
#         hidden: Optional[int] = None,
#         p_drop: float = 0.2,
#         cosine_scale_init: float = 16.0,      # s
#         learn_temperature: bool = True,       # learn T jointly, or set post-hoc later
#         init_temperature: float = 1.0,
#         use_logit_prior: bool = False,        # set True and call set_class_prior(...)
#     ):
#         super().__init__()
#         H = hidden or max(128, dim_in // 2)

#         # Pre-norm + tiny residual MLP
#         self.pre = nn.LayerNorm(dim_in)
#         self.fc1 = nn.Linear(dim_in, H)
#         self.act = nn.SiLU(inplace=True)
#         self.drop = nn.Dropout(p_drop)
#         self.fc2 = nn.Linear(H, dim_in)

#         # Cosine classifier: single weight vector → 1 logit
#         self.w = nn.Parameter(torch.empty(dim_in))
#         nn.init.normal_(self.w, std=0.02)

#         # Scale for cosine logit
#         self.log_s = nn.Parameter(torch.log(torch.tensor(cosine_scale_init)))

#         # Temperature scaling (applied to logits)
#         self.learn_temperature = learn_temperature
#         if learn_temperature:
#             self.log_T = nn.Parameter(torch.log(torch.tensor(init_temperature)))
#         else:
#             self.register_buffer("T_buffer", torch.tensor(init_temperature))
#             self.log_T = None

#         # Optional logit prior (bias) for imbalance
#         self.use_logit_prior = use_logit_prior
#         if use_logit_prior:
#             self.register_buffer("logit_prior", torch.tensor(0.0))
#         else:
#             self.register_buffer("logit_prior", torch.tensor(0.0))  # kept for simplicity

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Pre-norm + residual MLP
#         h = self.pre(x)
#         y = self.fc2(self.drop(self.act(self.fc1(h))))
#         h = h + y  # residual

#         # Cosine logit
#         h_norm = F.normalize(h, p=2, dim=1)           # (B,D)
#         w_norm = F.normalize(self.w, p=2, dim=0)      # (D,)
#         cos = torch.einsum('bd,d->b', h_norm, w_norm) # (B,)
#         s = torch.exp(self.log_s).clamp(1.0, 100.0)
#         logits = s * cos                               # (B,)

#         # Optional prior adjustment (class imbalance)
#         if self.use_logit_prior:
#             logits = logits + self.logit_prior

#         # Temperature scaling
#         T = (torch.exp(self.log_T) if self.log_T is not None else self.T_buffer).clamp(1e-3, 100.0)
#         logits = logits / T

#         return logits.unsqueeze(-1)  # (B,1)

#     @torch.no_grad()
#     def set_class_prior(self, pos_fraction: float, strength: float = 1.0):
#         """
#         Logit adjustment: add log(pi/(1-pi)) to logits (scaled by 'strength').
#         Call with validation-set prior or training prior.
#         """
#         pi = max(1e-6, min(1.0 - 1e-6, pos_fraction))
#         self.logit_prior.fill_(strength * math.log(pi / (1.0 - pi)))

#     @torch.no_grad()
#     def set_temperature(self, T_value: float):
#         """
#         If learn_temperature=False, you can still set T post-hoc from calibration.
#         """
#         if self.log_T is not None:
#             self.log_T.data = torch.log(torch.tensor(float(T_value), device=self.w.device))
#         else:
#             self.T_buffer.fill_(float(T_value))


# # -------- Optional losses you can plug in --------

# def bce_logits_loss(
#     logits: torch.Tensor, targets: torch.Tensor,
#     pos_weight: Optional[float] = None,
#     label_smoothing: float = 0.0,
#     reduction: str = "mean",
# ):
#     """
#     Standard BCEWithLogits with optional label smoothing and class pos_weight.
#     targets: {0,1}^(B,1)
#     """
#     if label_smoothing > 0:
#         t = targets.clamp(0,1).float()
#         eps = label_smoothing
#         t = t * (1 - eps) + 0.5 * eps
#     else:
#         t = targets.float()
#     return F.binary_cross_entropy_with_logits(
#         logits, t, reduction=reduction,
#         pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
#     )

# def focal_bce_with_logits(
#     logits: torch.Tensor, targets: torch.Tensor,
#     alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean",
# ):
#     """
#     Focal BCE for class imbalance/noisy labels.
#     """
#     p = torch.sigmoid(logits)
#     t = targets.float()
#     ce = F.binary_cross_entropy_with_logits(logits, t, reduction='none')
#     pt = p * t + (1 - p) * (1 - t)  # prob assigned to true class
#     w = (alpha * t + (1 - alpha) * (1 - t)) * ((1 - pt).clamp_min(1e-6) ** gamma)
#     loss = w * ce
#     return loss.mean() if reduction == "mean" else loss.sum()






# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional

# class RobustTriClassHead(nn.Module):
#     """
#     3-class head with:
#       - PreNorm + tiny residual MLP (SiLU)
#       - Cosine classifier (norm-invariant) with learnable scale s
#       - Temperature scaling (learned or post-hoc)
#       - Optional class-prior logit adjustment (for imbalance)
#       - Label smoothing support
#     """
#     def __init__(
#         self,
#         dim_in: int,
#         num_classes: int = 3,
#         hidden: Optional[int] = None,
#         p_drop: float = 0.2,
#         cosine_scale_init: float = 16.0,
#         learn_temperature: bool = True,
#         init_temperature: float = 1.0,
#         use_logit_prior: bool = False
#     ):
#         super().__init__()
#         assert num_classes >= 3
#         H = hidden or max(128, dim_in // 2)

#         # Pre-norm + tiny residual MLP
#         self.pre = nn.LayerNorm(dim_in)
#         self.fc1 = nn.Linear(dim_in, H)
#         self.act = nn.SiLU(inplace=True)
#         self.drop = nn.Dropout(p_drop)
#         self.fc2 = nn.Linear(H, dim_in)

#         # Cosine classifier weights: W ∈ R^{D×K}
#         self.W = nn.Parameter(torch.empty(dim_in, num_classes))
#         nn.init.normal_(self.W, std=0.02)

#         # Scale for cosine logits
#         self.log_s = nn.Parameter(torch.log(torch.tensor(cosine_scale_init)))

#         # Temperature (log-space)
#         self.learn_temperature = learn_temperature
#         if learn_temperature:
#             self.log_T = nn.Parameter(torch.log(torch.tensor(init_temperature)))
#         else:
#             self.register_buffer("T_buffer", torch.tensor(init_temperature))
#             self.log_T = None

#         # Optional class-prior adjustment
#         self.use_logit_prior = use_logit_prior
#         self.register_buffer("logit_prior", torch.zeros(num_classes))

#     def forward(self, x: torch.Tensor, return_probs: bool = False):
#         # Pre-norm + residual MLP
#         h = self.pre(x)
#         y = self.fc2(self.drop(self.act(self.fc1(h))))
#         h = h + y  # residual

#         # Cosine logits
#         h = F.normalize(h, p=2, dim=1)                  # (B,D)
#         W = F.normalize(self.W, p=2, dim=0)             # (D,K)
#         cos = torch.matmul(h, W)                         # (B,K)
#         s = torch.exp(self.log_s).clamp(1.0, 100.0)
#         logits = s * cos                                 # (B,K)

#         if self.use_logit_prior:
#             logits = logits + self.logit_prior          # broadcast (B,K)

#         T = (torch.exp(self.log_T) if self.log_T is not None else self.T_buffer).clamp(1e-3, 100.0)
#         logits = logits / T

#         if return_probs:
#             return logits, F.softmax(logits, dim=-1)
#         return logits

#     @torch.no_grad()
#     def set_class_priors(self, priors: torch.Tensor, strength: float = 1.0):
#         """
#         priors: shape (K,), sum to 1. Adds strength * log(pi_k) to logits.
#         """
#         pi = priors.clamp_min(1e-6)
#         self.logit_prior.copy_(strength * torch.log(pi))

# Loss helpers
def xe_with_label_smoothing(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.05):
    """
    targets: int64 class indices, shape (B,)
    """
    if smoothing <= 0:
        return F.cross_entropy(logits, targets)
    n_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits).fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    logp = F.log_softmax(logits, dim=-1)
    return torch.mean(torch.sum(-true_dist * logp, dim=-1))







# class RobustTriClassHead(nn.Module):
#     """
#     Cosine classifier + EMA prototypes + learnable logit fusion.
#     Use: logits, aux = head(z, y=labels)   # y optional; required only if you call update_prototypes(...)
#     """
#     def __init__(self, dim_in: int, num_classes: int = 3,
#                  p_drop: float = 0.2,
#                  use_prototypes: bool = True,
#                  proto_momentum: float = 0.99,
#                  init_scale: float = 10.0,
#                  init_scale_proto: float = 10.0,
#                  fuse_init: float = -1.5  # sigmoid -> ~0.18
#                  ):
#         super().__init__()
#         self.C = num_classes
#         self.D = dim_in
#         self.use_prototypes = use_prototypes
#         self.m = proto_momentum

#         # Linear cosine classifier (weights are learnable, normalized on the fly)
#         self.W = nn.Parameter(torch.randn(self.C, self.D) * 0.02)
#         self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale)))

#         # Prototype branch (buffers, updated via EMA)
#         if self.use_prototypes:
#             self.register_buffer("prototypes", F.normalize(torch.randn(self.C, self.D), dim=1))
#             self.register_buffer("proto_counts", torch.zeros(self.C))  # for warmup masking
#             self.log_scale_proto = nn.Parameter(torch.log(torch.tensor(init_scale_proto)))
#             self.fuse_logit = nn.Parameter(torch.tensor(fuse_init))  # sigmoid->lambda in [0,1]

#         self.dropout = nn.Dropout(p_drop)

#     @staticmethod
#     def _cos_logits(z, W, scale):
#         # z: (B,D), W: (C,D)
#         z = F.normalize(z, dim=1)
#         W = F.normalize(W, dim=1)
#         return scale * (z @ W.t())  # (B,C)

#     @torch.no_grad()
#     def update_prototypes(self, z: torch.Tensor, y: torch.Tensor):
#         """
#         Update EMA prototypes using CURRENT minibatch embeddings (already from head input).
#         Call this in training loop after forward(), before loss.backward().
#         """
#         if not self.use_prototypes:
#             return
#         z = F.normalize(z, dim=1)
#         for c in range(self.C):
#             idx = (y == c)
#             if idx.any():
#                 batch_mean = z[idx].mean(dim=0)
#                 self.prototypes[c] = F.normalize(self.m * self.prototypes[c] + (1 - self.m) * batch_mean, dim=0)
#                 self.proto_counts[c] += idx.sum()

#     def forward(self, z: torch.Tensor, y: torch.Tensor | None = None):
#         # z: (B, D)
#         z = self.dropout(z)

#         s = self.log_scale.exp()
#         logits_lin = self._cos_logits(z, self.W, s)  # (B,C)

#         aux = {"scale": float(s.detach().cpu())}

#         if not self.use_prototypes:
#             return logits_lin, aux

#         sp = self.log_scale_proto.exp()
#         logits_proto = self._cos_logits(z, self.prototypes, sp)

#         # Mask proto logits for classes never seen yet (counts==0) to avoid garbage early on
#         seen = (self.proto_counts > 0).float()  # (C,)
#         if (seen == 0).any():
#             mask = seen.unsqueeze(0)  # (1,C)
#             logits_proto = logits_proto * mask + logits_lin * (1 - mask)

#         lam = torch.sigmoid(self.fuse_logit)  # (scalar in 0..1)
#         logits = (1 - lam) * logits_lin + lam * logits_proto

#         aux.update({
#             "scale_proto": float(sp.detach().cpu()),
#             "lambda": float(lam.detach().cpu())
#         })
#         return logits, aux


# -------- Losses --------

class LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes: int, eps: float = 0.05, weight: torch.Tensor | None = None):
        super().__init__()
        self.C = num_classes
        self.eps = eps
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        logp = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(self.eps / (self.C - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        if self.weight is not None:
            loss = -(true_dist * logp) * self.weight.unsqueeze(0)
            return loss.sum(dim=1).mean()
        return -(true_dist * logp).sum(dim=1).mean()


class ClassBalancedFocal(nn.Module):
    """
    Class-balanced focal loss (Cui et al.), good for imbalance.
    Provide counts via 'counts' tensor of shape (C,)
    """
    def __init__(self, counts: torch.Tensor, beta: float = 0.999, gamma: float = 2.0):
        super().__init__()
        C = counts.numel()
        effective_num = 1.0 - counts.float().clamp_min(1) * (1 - beta)
        weights = (1 - beta) / effective_num
        weights = weights / weights.sum() * C
        self.register_buffer("alpha", weights)  # (C,)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        w = self.alpha.gather(0, target)
        focal = ((1 - pt) ** self.gamma) * (-logp.gather(1, target.unsqueeze(1)).squeeze(1))
        return (w * focal).mean()



