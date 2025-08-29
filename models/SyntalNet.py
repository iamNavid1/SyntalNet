import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import List, Tuple, Optional, Dict

from models.base_model import BaseModel
from models.squeeze_excite import CrossSE
from models.fusion import GPSFusion, GLRFusion
from models.classifier import ClassificationHead
from models.partial import PartialConv2d, PartialAvgPool2d, PartialGeM
from models.utils import MLP, ChannelLayerNorm2d, _preserves_spatial, _is_same_like


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

        self.prenorm = ChannelLayerNorm2d(in_channels)
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
        preserves = _preserves_spatial(kernel_size, stride, padding, dilation)
        need_mask_update = (stride != (1, 1))
        ceil = _is_same_like(kernel_size, stride, padding, dilation)

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


class Branch(nn.Module):
    def __init__(
            self,
            in_ch_emb: int,
            in_ch_feat: int,
            dim_emb: int,
            dim_feat: int,
            use_emb: bool = True,
            use_feat: bool = True,
            num_blocks: int = 3,
            upsample_emb: bool = False,
            shared_dim: int = 128,
            shared_out_ch: int = 32,
            ch_expansion: List[int] = [2, 2],
            ks_emb: List[int | Tuple[int, int]]  = [(5,1), (5,1), (5,1)],
            ks_feat: List[int | Tuple[int, int]] = [5, 5, 5],
            st_emb: List[int | Tuple[int, int]]  = [(1,2), (1,2), (1,2)],
            st_feat: List[int | Tuple[int, int]] = [1, 1, 1],
            pd_emb: List[int | Tuple[int, int]]  = [(2,0), (2,0), (2,0)],
            pd_feat: List[int | Tuple[int, int]] = [2, 2, 2],
            dl_emb: List[int | Tuple[int, int]]  = [1, 1, 1],
            dl_feat: List[int | Tuple[int, int]] = [1, 1, 1],
            p_drop_block: float = 0.05,
            activation_block: str = nn.SiLU,
            residual: bool = True,
            cross_se: bool = True,
            p_drop_proj: float = 0.05,
            activation_proj: str = nn.GELU,
    ):
        super().__init__()

        if not all(num_blocks == len(lst) for lst in [ks_emb, ks_feat, st_emb, st_feat, pd_emb, pd_feat, dl_emb, dl_feat]):
            raise ValueError(f"Expected length {num_blocks} for CNN backbone inputs")

        if not (use_emb or use_feat):
            raise ValueError("At least one of use_emb or use_feat must be True")

        self.use_emb = use_emb
        self.use_feat = use_feat

        if upsample_emb and use_emb:
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

        if use_emb:
            self.blocks_emb = nn.ModuleList([
                CNNBackbone(
                    in_ch_emb if i == 0 else shared_out_ch, shared_out_ch,
                    ks_emb[i], st_emb[i], pd_emb[i], dl_emb[i],
                    p_drop_block, activation_block, 
                    residual, cross_se,
                ) for i in range(num_blocks)
            ])
        else:
            self.blocks_emb = None

        if use_feat:
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
        else:
            self.blocks_feat = None
            self.feat_projector = None

        if use_emb and use_feat:
            self.mc_fusion = GPSFusion(C_mod=shared_out_ch, C_exp=ch_expansion)  # intramodality multi-channel fusion
            self.pool = None
            self.drop = None
            self.fc = None
        else:
            self.mc_fusion = None
            self.pool = PartialGeM()
            self.drop = nn.Dropout(p_drop_proj)
            self.fc = nn.Linear(shared_out_ch, 2 * shared_out_ch * ch_expansion[0] * ch_expansion[1])

    def cnn_block_forward(self, x, m, blocks):
        out = x
        for block in blocks:
            out, m = block(out, m)
        return out, m

    def forward(
            self,
            x_feat: Optional[List[torch.Tensor]] = None,  # (B, P, T, F_f_i)
            m_feat: Optional[List[torch.Tensor]] = None,  # (B, P, T, F_f_i)
            x_emb: Optional[torch.Tensor] = None,         # (B, P, (T)(T_e), F_e)
            m_emb: Optional[torch.Tensor] = None,         # (B, P, (T)(T_e), F_e)
    ) -> torch.Tensor:
        
        B, P, _, _ = (x_emb.shape if x_emb is not None
                        else x_feat[0].shape)

        if self.use_feat:
            assert x_feat is not None, "Feature inputs required when use_feat=True"
            x_feat = torch.cat(x_feat, dim=-1).unsqueeze(2)
            m_feat = torch.cat(m_feat, dim=-1).unsqueeze(2)

            z_feat, m_feat = self.cnn_block_forward(x_feat, m_feat, self.blocks_feat)
            z_feat = z_feat.view(B * P, z_feat.shape[-3], z_feat.shape[-2], z_feat.shape[-1])
            z_feat = self.feat_projector(z_feat)
            m_feat = m_feat.view(B * P, m_feat.shape[-3], m_feat.shape[-2], m_feat.shape[-1])
            m_feat = (m_feat.sum(dim=-1, keepdim=True) > 0).expand(B * P, m_feat.shape[-3], m_feat.shape[-2], z_feat.shape[-1])

        if self.use_emb:
            assert x_emb is not None, "Embedding inputs required when use_emb=True"
            if self.emb_upsampler:
                x_emb = x_emb.view(B * P, x_emb.shape[-2], x_emb.shape[-1]).transpose(1, 2)
                x_emb = self.emb_upsampler(x_emb)
                x_emb = x_emb.view(B, P, x_emb.shape[-2], x_emb.shape[-1]).transpose(2, 3)
            x_emb = x_emb.unsqueeze(2)
            m_emb = m_emb.unsqueeze(2)

            z_emb, m_emb = self.cnn_block_forward(x_emb, m_emb, self.blocks_emb)
            z_emb = z_emb.view(B * P, z_emb.shape[-3], z_emb.shape[-2], z_emb.shape[-1])
            m_emb = m_emb.view(B * P, m_emb.shape[-3], m_emb.shape[-2], m_emb.shape[-1])
        
        if self.use_feat and self.use_emb:
            z_branch = self.mc_fusion(z_feat, z_emb, m_feat, m_emb)
        else:
            z, m = (z_feat, m_feat) if self.use_feat else (z_emb, m_emb)
            z_branch = self.pool(z, m).flatten(1)
            z_branch = self.drop(z_branch)
            z_branch = self.fc(z_branch)

        return z_branch


class SyntalNet(BaseModel):
    def __init__(
            self,
            branches: List[str]         = ['Videokinetic', 'Dialogue', 'Acoustic'],
            dim_out: int                = 256,
            in_ch_emb: List[int]        = [1, 1, 1],
            in_ch_feat: List[int]       = [1, 1, 1],
            dim_emb: List[int]          = [1024, 1024, 512],
            dim_feat: List[int]         = [394, 12, 12],
            use_emb: bool               = True,
            use_feat: bool              = True,
            num_blocks: List[int]       = [3, 3, 3],
            upsample_emb: bool          = [True, False, False],
            shared_dim: int             = 128,
            shared_out_ch: int          = 32,
            ch_expansion: List[int]     = [2, 2],
            stride_emb: List[List[int]] = [[(1,2), (1,2), (1,2)], [(1,2), (1,2), (1,2)], [(1,1), (1,2), (1,2)]],
            branch_residual: bool       = True,
            cross_se: bool              = True,
            ind_cls_heads: Tuple[str]   = (),
            grp_cls_heads: Tuple[str]   = (),
            use_prototypes: bool        = True,
            proto_warmup_epochs: int    = 5,
            cls_residual: bool          = True,
            cls_head_type: str          = "cosine",
    ):
        super().__init__()

        assert all(branch in ['Videokinetic', 'Dialogue', 'Acoustic'] for branch in branches), \
            "Branches must be one of 'Videokinetic', 'Dialogue', or 'Acoustic'"

        assert len(branches) == len(in_ch_emb) == len(in_ch_feat) == len(dim_emb) == len(dim_feat) == len(num_blocks) == len(upsample_emb) == len(stride_emb), \
            "All input lists (branches, in_ch_emb, in_ch_feat, dim_emb, dim_feat, num_blocks, upsample_emb) must have the same length"
            
        assert all(len(s) == nb for s, nb in zip(stride_emb, num_blocks)), \
            "Each stride_emb[i] must have length equal to num_blocks[i]"

        self.branches = nn.ModuleDict()
        for i, branch in enumerate(branches):
            self.branches[branch] = Branch(
                in_ch_emb     = in_ch_emb[i],
                in_ch_feat    = in_ch_feat[i],
                dim_emb       = dim_emb[i],
                dim_feat      = dim_feat[i],
                use_emb       = use_emb,
                use_feat      = use_feat,
                num_blocks    = num_blocks[i],
                upsample_emb  = upsample_emb[i],
                shared_dim    = shared_dim,
                shared_out_ch = shared_out_ch,
                ch_expansion  = ch_expansion,
                st_emb        = stride_emb[i],
                residual      = branch_residual,
                cross_se      = cross_se,
            )

        if len(branches) > 1:
            self.mm_fusion = GLRFusion(
                dims_mod     = [2 * shared_out_ch * ch_expansion[0] * ch_expansion[1] for _ in range(len(branches))],
                dim_hidden   = shared_out_ch * ch_expansion[0] * ch_expansion[1],
                rank_pair    = shared_out_ch,
                alloc_hidden = 2 * shared_out_ch,
                dim_out      = dim_out,
            )
        else:
            self.mm_fusion = None

        self.individual_classifier = None
        self.group_classifier = None

        if ind_cls_heads:
            self.individual_classifier = ClassificationHead(
                dim             = dim_out,
                head_names      = list(ind_cls_heads),
                use_prototypes  = use_prototypes,
                residual        = cls_residual,
                warmup_epochs   = proto_warmup_epochs,
                classifier_type = cls_head_type,
            )

        if grp_cls_heads:
            self.group_classifier = ClassificationHead(
                dim             = dim_out,  # will concat all people: dim -> 3*dim
                head_names      = list(grp_cls_heads),
                trunk_hidden    = dim_out,
                adapter_hidden  = dim_out,
                use_prototypes  = use_prototypes,
                residual        = cls_residual,
                warmup_epochs   = proto_warmup_epochs,
                group_wise      = True,
                classifier_type = cls_head_type,
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

        if len(self.branches) > 1:
            z = self.mm_fusion(z_branch)
        else:
            z = z_branch[0]

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


