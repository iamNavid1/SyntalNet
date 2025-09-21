import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from typing import List, Tuple, Dict, Optional, Any, Type

from models.base_model import BaseModel
from models.squeeze_excite import CrossSE
from models.fusion import GPSFusion, GLRFusion
from models.classifier import ClassificationHead
from models.encoder import CNXv2Block, FMixLowRank, StageTransition
from models.partial import PartialConv2d, PartialAvgPool2d, PartialGeM
from models.utils import MLP, ChannelLayerNorm2d, _preserves_spatial, _is_same_like


class EncoderStage(nn.Module):
    """
    One stage with:
      - A list of depth CNXv2Blocks
      - Optional PartialGeM over T
      - Low-rank Feature Mixer over F w/ w/o dimension reduction
    """
    def __init__(
        self,
        dim: int,
        C: int,
        depth: int,
        ks: List[int | Tuple[int, int]],
        pd: List[int | Tuple[int, int]],
        st: List[int | Tuple[int, int]],
        dl: List[int | Tuple[int, int]],
        expansion: int,
        last_stage: bool,
        act: nn.Module,
        p_drop: float,
        p_droppath: List[float],
        pool: Optional[int | Tuple[int, int]],
        fmix_rank: Optional[int] = None,
    ):
        super().__init__()

        def _broadcast(v: Any, name: str):
            if isinstance(v, list):
                if len(v) == 1:
                    return [v[0]] * depth
                assert len(v) == depth, f"{name} must have length 1 or depth={depth}"
                return v
            return [v] * depth

        ks         = _broadcast(ks, "ks")
        pd         = _broadcast(pd, "pd")
        st         = _broadcast(st, "st")
        dl         = _broadcast(dl, "dl")
        p_droppath = _broadcast(p_droppath, "p_droppath")

        self.blocks = nn.ModuleList([
            CNXv2Block(
                C=C, k=ks[i], p=pd[i], s=st[i], d=dl[i],
                expansion=expansion, act=act,
                p_drop=p_drop, p_droppath=p_droppath[i],
            ) for i in range(depth)
        ])

        self.temporal_pool = None
        if pool is not None:
            p_t, p_f = _pair(pool)
        else:
            p_t = None
            p_f = 1

        self.temporal_pool = None
        if p_t is not None and p_t > 1:
            self.temporal_pool = PartialGeM((p_t, 1), global_pool=False, return_mask=True)
        if last_stage:
            self.fmix = FMixLowRank(F_in=dim, F_out=dim//p_f, rank=fmix_rank, act=act, pre_norm=True)
        else:
            self.fmix = FMixLowRank(F_in=dim, F_out=dim//p_f, rank=fmix_rank, pre_norm=(p_f != 1))

    def forward(self, x: torch.Tensor, m:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x, m = blk(x, m)
        if self.temporal_pool is not None:
            x, m = self.temporal_pool(x, m)
        x, m = self.fmix(x, m)
        return x, m


class EncoderBackbone(nn.Module):
    """
    Multi-stage encoder:
      - optional stem StageTransition to first stage width
      - per stage: EncoderStage -> optional CrossSE -> optional StageTransition to next stage
    """
    def __init__(
        self,
        dim: int,
        in_channels: int,
        num_stages: int,
        stage_depths:    List[int],
        stage_channels:  List[int],
        stage_kernels:   List[List[int | Tuple[int, int]]],
        stage_pads:      List[List[int | Tuple[int, int]]],
        stage_strides:   List[List[int | Tuple[int, int]]],
        stage_dilations: List[List[int | Tuple[int, int]]],
        stage_pool:      List[Optional[int | Tuple[int, int]]],
        expansion:       int = 2,
        act:             nn.Module = nn.GELU,
        p_drop:          float = .05,
        p_droppath:      float = .05,  # max DropPath
        fmix_rank:       Optional[int] = None,
        apply_mp_fusion: bool = True,  # multi-person feature map fusion
    ):
        super().__init__()

        def _broadcast(lst, name):
            if len(lst) == 1:
                return [lst[0]] * num_stages
            assert len(lst) == num_stages, f"Expected length {num_stages} for {name}, got {len(lst)}"
            return lst

        stage_channels  = _broadcast(stage_channels, "stage_channels")
        stage_depths    = _broadcast(stage_depths, "stage_depths")
        stage_kernels   = _broadcast(stage_kernels, "stage_kernels")
        stage_pads      = _broadcast(stage_pads, "stage_pads")
        stage_strides   = _broadcast(stage_strides, "stage_strides")
        stage_dilations = _broadcast(stage_dilations, "stage_dilations")
        stage_pool      = _broadcast(stage_pool, "stage_pool")
        self.stage_pool = stage_pool

        self.num_stages = num_stages
        self.apply_mp_fusion = apply_mp_fusion

        first_C = stage_channels[0]
        self.stem = StageTransition(in_channels, first_C, norm=True) \
            if in_channels != first_C \
            else None

        total_blocks = sum(stage_depths)
        dp_rates = torch.linspace(0.0, p_droppath, total_blocks).tolist()

        self.stages = nn.ModuleList()
        self.cross  = nn.ModuleList()
        self.trans  = nn.ModuleList()

        cur = 0
        cur_F = dim
        for i in range(num_stages):
            depth_i = stage_depths[i]
            dp_i = dp_rates[cur : cur+depth_i]
            cur += depth_i

            stage_i = EncoderStage(
                dim=cur_F,
                C=stage_channels[i],
                depth=depth_i,
                ks=stage_kernels[i],
                pd=stage_pads[i],
                st=stage_strides[i],
                dl=stage_dilations[i],
                expansion=expansion,
                last_stage=(i==num_stages-1),
                act=act,
                p_drop=p_drop,
                p_droppath=dp_i,
                pool=stage_pool[i],
                fmix_rank=fmix_rank,
            )
            self.stages.append(stage_i)

            if apply_mp_fusion:
                self.cross.append(CrossSE(num_channels=stage_channels[i], dropout_p=p_drop, norm=(i==num_stages-1)))
            else:
                self.cross.append(nn.Identity())

            if i < num_stages - 1:
                if stage_channels[i] != stage_channels[i + 1]:
                    self.trans.append(StageTransition(stage_channels[i], stage_channels[i + 1], norm=True))
                else:
                    self.trans.append(nn.Identity())
            else:
                self.trans.append(nn.Identity())

            pool_i = stage_pool[i]
            if pool_i is not None:
                _, p_f_i = _pair(pool_i)
            else:
                p_f_i = 1
            cur_F = cur_F // p_f_i

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 5 and m.ndim == 5, "x and m must be 5D: (B,P,C,T,F) and/or (B,P,1,T,F)"

        B, P, C, T, F = x.shape
        x = x.view(B * P, C, T, F)
        m = m.view(B * P, 1, T, F)

        if self.stem is not None:
            x, m = self.stem(x, m)

        for stage, fusion, transition in zip(self.stages, self.cross, self.trans):
            x, m = stage(x, m)
            if not isinstance(fusion, nn.Identity):
                x = fusion(x)
            if not isinstance(transition, nn.Identity):
                x, m = transition(x, m)

        x = x.view(B, P, x.shape[-3], x.shape[-2], x.shape[-1])
        m = m.view(B, P, m.shape[-3], m.shape[-2], m.shape[-1])

        return x, m


class Branch(nn.Module):
    def __init__(
            self,
            in_ch_emb: int,
            in_ch_feat: int,
            dim_emb: int,
            dim_feat: int,
            use_emb: bool,
            use_feat: bool,
            num_stages: int,
            stage_depths: List[int],
            stage_channels: List[int],
            shared_dim: int = 64,
            upsample_emb: bool = False,
            fus_pool_schd: List[int | Tuple[int, int]] = [(4,4), (3,4)],
            fus_ch_schd: Optional[List[int]] = [32, 32, 16],
            fus_trgt_grd: Tuple[int,int] = [4, 4],
            ks_emb: List[List[int | Tuple[int, int]]]  = [(5,1)],
            ks_feat: List[List[int | Tuple[int, int]]] = [(5,1)],
            st_emb: List[List[int | Tuple[int, int]]]  = [1],
            st_feat: List[List[int | Tuple[int, int]]] = [1],
            pd_emb: List[List[int | Tuple[int, int]]]  = [(2,0)],
            pd_feat: List[List[int | Tuple[int, int]]] = [(2,0)],
            dl_emb: List[List[int | Tuple[int, int]]]  = [1],
            dl_feat: List[List[int | Tuple[int, int]]] = [1],
            pool_emb: List[int | Tuple[int, int]] = [1],
            pool_feat: List[int | Tuple[int, int]] = [1],
            p_drop_block: float = 0.05,
            p_droppath_block: float = 0.05,
            activation_block: Type[nn.Module] = nn.GELU,
            p_drop_proj: float = 0.1,
            activation_proj: Type[nn.Module] = nn.GELU,
            fmix_rank_emb: Optional[int] = None,
            fmix_rank_feat: Optional[int] = None,
            cross_se: bool = True,
            p_drop_mod_s: Optional[float | Dict[str, float]] = None,
            p_drop_mod_e: Optional[float | Dict[str, float]] = None,
    ):
        super().__init__()

        if not (use_emb or use_feat):
            raise ValueError("At least one of use_emb or use_feat must be True")

        self.use_emb = use_emb
        self.use_feat = use_feat

        # if use_emb:
        #     self.anchor_emb = nn.Parameter(torch.zeros(1, 1, 1, dim_emb))
        #     nn.init.trunc_normal_(self.anchor_emb, std=0.02)
        # else:
        #     self.anchor_emb = None
        # if use_feat:
        #     self.anchor_feat = nn.Parameter(torch.zeros(1, 1, 1, dim_feat))
        #     nn.init.trunc_normal_(self.anchor_feat, std=0.02)
        # else:
        #     self.anchor_feat = None

        if upsample_emb and use_emb:
            self.emb_upsampler = nn.ConvTranspose1d(
                in_channels = dim_emb,
                out_channels = dim_emb,
                kernel_size = 8,
                stride = 3,
                padding = 0,
                dilation=3,
                output_padding = 0,
                groups=dim_emb,
                bias=False,
            )
        else:
            self.emb_upsampler = None

        if use_emb:
            self.encoder_emb = EncoderBackbone(
                dim=dim_emb,
                in_channels=in_ch_emb,
                num_stages=num_stages,
                stage_depths=stage_depths,
                stage_channels=stage_channels,
                stage_kernels=ks_emb,
                stage_pads=pd_emb,
                stage_strides=st_emb,
                stage_dilations=dl_emb,
                stage_pool=pool_emb,
                act=activation_block,
                p_drop=p_drop_block,
                p_droppath=p_droppath_block,
                fmix_rank=fmix_rank_emb,
                apply_mp_fusion=cross_se,
            )
            div = 1
            for pool in self.encoder_emb.stage_pool:
                if pool is not None:
                    div *= _pair(pool)[1]
            self.projector_emb = MLP(
                in_features=dim_emb//div, hidden_features=32, out_features=shared_dim,
                p_drop=p_drop_proj, act_layer=activation_proj
            )
        else:
            self.encoder_emb = None
            self.projector_emb = None

        if use_feat:
            self.encoder_feat = EncoderBackbone(
                dim=dim_feat,
                in_channels=in_ch_emb,
                num_stages=num_stages,
                stage_depths=stage_depths,
                stage_channels=stage_channels,
                stage_kernels=ks_feat,
                stage_pads=pd_feat,
                stage_strides=st_feat,
                stage_dilations=dl_feat,
                stage_pool=pool_feat,
                act=activation_block,
                p_drop=p_drop_block,
                p_droppath=p_droppath_block,
                fmix_rank=fmix_rank_feat,
                apply_mp_fusion=cross_se,
            )
            div = 1
            for pool in self.encoder_feat.stage_pool:
                if pool is not None:
                    div *= _pair(pool)[1]
            self.projector_feat = MLP(
                in_features=dim_feat//div, hidden_features=32, out_features=shared_dim, 
                p_drop=p_drop_proj, act_layer=activation_proj
            )
        else:
            self.blocks_feat = None
            self.projector_feat = None

        # intra-modality fusion
        if use_emb and use_feat:
            self.mc_fusion = GPSFusion(
                C_mod            = stage_channels[-1], 
                F_mod            = shared_dim,
                pool_schedule    = fus_pool_schd,
                channel_schedule = fus_ch_schd,
                target_grid      = fus_trgt_grd,
                p_drop_mod_s     = p_drop_mod_s,
                p_drop_mod_e     = p_drop_mod_e,
            )
            self.pool = None
            self.drop = None
            self.fc = None
        else:
            self.mc_fusion = None
            self.pool = PartialGeM(global_pool=True)
            self.drop = nn.Dropout(p_drop_proj)
            self.fc = nn.Linear(stage_channels[-1], fus_ch_schd[-1] * int(np.prod(fus_trgt_grd)))

    def forward(
            self,
            x_feat: Optional[List[torch.Tensor]] = None,  # (B, P, T, F_f_i)
            m_feat: Optional[List[torch.Tensor]] = None,  # (B, P, T, F_f_i)
            x_emb:  Optional[torch.Tensor]       = None,  # (B, P, (T)(T_e), F_e)
            m_emb:  Optional[torch.Tensor]       = None,  # (B, P, (T)(T_e), F_e)
            epoch:  Optional[int]                = None,
    ) -> torch.Tensor:
        
        B, P = (x_emb.shape[:2] if x_emb is not None
                        else x_feat[0].shape[:2])

        if self.use_feat:
            assert x_feat is not None, "Feature inputs required when use_feat=True"
            x_feat = torch.cat(x_feat, dim=-1)
            m_feat = torch.cat(m_feat, dim=-1)
            # anchor = self.anchor_feat.expand(B, P, -1, -1)
            # mask_anchor = torch.ones(B, P, 1, anchor.shape[-1], device=m_feat.device, dtype=m_feat.dtype)
            # x_feat = torch.cat([anchor, x_feat], dim=2).unsqueeze(2)
            # m_feat = torch.cat([mask_anchor, m_feat], dim=2).unsqueeze(2)
            x_feat = x_feat.unsqueeze(2)
            m_feat = m_feat.unsqueeze(2)

            zf, mf = self.encoder_feat(x_feat, m_feat)
            zf = zf.view(B * P, zf.shape[-3], zf.shape[-2], zf.shape[-1])
            zf = self.projector_feat(zf)
            mf = mf.view(B * P, mf.shape[-3], mf.shape[-2], mf.shape[-1])
            mf = (mf.sum(dim=-1, keepdim=True) > 0).expand(B * P, mf.shape[-3], mf.shape[-2], zf.shape[-1])

        if self.use_emb:
            assert x_emb is not None, "Embedding inputs required when use_emb=True"
            if self.emb_upsampler:
                x_emb = x_emb.view(B * P, x_emb.shape[-2], x_emb.shape[-1]).transpose(1, 2)
                x_emb = self.emb_upsampler(x_emb)
                x_emb = x_emb.view(B, P, x_emb.shape[-2], x_emb.shape[-1]).transpose(2, 3)
            # anchor = self.anchor_emb.expand(B, P, -1, -1)
            # mask_anchor = torch.ones(B, P, 1, anchor.shape[-1], device=m_emb.device, dtype=m_emb.dtype)
            # x_emb = torch.cat([anchor, x_emb], dim=2).unsqueeze(2)
            # m_emb = torch.cat([mask_anchor, m_emb], dim=2).unsqueeze(2)
            x_emb = x_emb.unsqueeze(2)
            m_emb = m_emb.unsqueeze(2)

            ze, me = self.encoder_emb(x_emb, m_emb)
            ze = ze.view(B * P, ze.shape[-3], ze.shape[-2], ze.shape[-1])
            ze = self.projector_emb(ze)
            me = me.view(B * P, me.shape[-3], me.shape[-2], me.shape[-1])
            me = (me.sum(dim=-1, keepdim=True) > 0).expand(B * P, me.shape[-3], me.shape[-2], ze.shape[-1])
        
        if self.use_feat and self.use_emb:
            z_branch = self.mc_fusion(zf, ze, mf, me, epoch)
        else:
            z, m = (zf, mf) if self.use_feat else (ze, me)
            z_branch = self.pool(z, m).flatten(1)
            z_branch = self.drop(z_branch)
            z_branch = self.fc(z_branch)

        return z_branch


class SyntalNet(BaseModel):
    def __init__(
        self,
        branches: Optional[List[str]] = None,
        in_ch_emb: Optional[List[int]] = None,
        in_ch_feat: Optional[List[int]] = None,
        dim_emb: Optional[List[int]] = None,
        dim_feat: Optional[List[int]] = None,
        use_emb: bool = True,
        use_feat: bool = True,
        num_stages: Optional[List[int]] = None,
        stage_depths: Optional[List[List[int]]] = None,
        stage_channels: Optional[List[List[int]]] = None,
        shared_dim: int = 64,
        upsample_emb: Optional[List[bool]] = None,
        fus_pool_schd: Optional[List[List[int | Tuple[int, int]]]] = None,
        fus_ch_schd: Optional[List[List[int]]] = None,
        fus_trgt_grd: Optional[List[Tuple[int,int]]] = None,
        ks_emb: Optional[List[List[List[int | Tuple[int, int]]]]] = None,
        ks_feat: Optional[List[List[List[int | Tuple[int, int]]]]] = None,
        st_emb: Optional[List[List[List[int | Tuple[int, int]]]]] = None,
        st_feat: Optional[List[List[List[int | Tuple[int, int]]]]] = None,
        pd_emb: Optional[List[List[List[int | Tuple[int, int]]]]] = None,
        pd_feat: Optional[List[List[List[int | Tuple[int, int]]]]] = None,
        dl_emb: Optional[List[List[List[int | Tuple[int, int]]]]] = None,
        dl_feat: Optional[List[List[List[int | Tuple[int, int]]]]] = None,
        pool_emb: Optional[List[List[int | Tuple[int, int]]]] = None,
        pool_feat: Optional[List[List[int | Tuple[int, int]]]] = None,
        p_drop_block: float = 0.1,
        p_droppath_block: float = 0.0,
        p_drop_proj: float = 0.1,
        fmix_rank_emb: Optional[int] = None,
        fmix_rank_feat: Optional[int] = None,
        cross_se: bool = True,
        ind_cls_heads: Tuple[str, ...] = (),
        grp_cls_heads: Tuple[str, ...] = (),
        use_prototypes: bool = True,
        trunk_phi_h: Optional[int] = 128,
        trunk_phi: Optional[int] = 96,
        trunk_hidden: Optional[int] = 96,
        lora_adapter: bool = True,
        adapter_hidden: Optional[int] = 48,
        proto_warmup_epochs: int = 0,
        cls_residual: bool = True,
        cls_head_type: str = "cosine",
        p_drop_mod_s: Optional[float | Dict[str, float]] = None,
        p_drop_mod_e: Optional[float | Dict[str, float]] = None,
    ):
        super().__init__()

        # ----------------------- defaults -----------------------
        branches = branches or ['Videokinetic', 'Dialogue', 'Acoustic']
        valid_branches = {'Videokinetic', 'Dialogue', 'Acoustic'}
        if not set(branches).issubset(valid_branches):
            raise ValueError("branches must be drawn from {'Videokinetic','Dialogue','Acoustic'}")
        B = len(branches)

        _per_branch_defaults = {
            'Videokinetic': dict(dim_emb=1024, dim_feat=394, upsample=True,
                                 num_stages=3, stage_depths=[2], 
                                 stage_channels=[8, 16, 32],
                                 pool_emb=[(2, 2)], pool_feat=[(2, 1)], 
                                 r_emb=16, r_feat=8,
                                 ks=[[(5,1)], [(7,1)], [(5,1), (7,1)]],
                                 pd=[[(2,0)], [(3,0), (6,0)], [(8,0), (3,0)]],
                                 dl=[[1], [1, (2,1)], [(4,1), 1]],
                                 fus_pool_schd=[(3,2), (4,8)], fus_ch_schd=[32, 32, 16], fus_trgt_grd=(4, 4)),
            'Dialogue'    : dict(dim_emb=1024, dim_feat=12,  upsample=False,
                                 num_stages=3, stage_depths=[2], 
                                 stage_channels=[8, 16, 32],
                                 pool_emb=[(1, 2)], pool_feat=[None], 
                                 r_emb=16, r_feat=8,
                                 ks=[[(5,1)]], pd=[[(2, 0)]], dl=[[1]],
                                 fus_pool_schd=[(3,2), (4,8)], fus_ch_schd=[32, 32, 16], fus_trgt_grd=(4, 4)),
            'Acoustic'    : dict(dim_emb=512,  dim_feat=12,  upsample=False,
                                 num_stages=3, stage_depths=[2],
                                 stage_channels=[8, 16, 32],
                                 pool_emb=[(2,1), (2, 2), (2, 2)], pool_feat=[(2, 1)],
                                 r_emb=16, r_feat=8,
                                 ks=[[(5,1)], [(7,1)], [(5,1), (7,1)]],
                                 pd=[[(2,0)], [(3,0), (6,0)], [(8,0), (3,0)]],
                                 dl=[[1], [1, (2,1)], [(4,1), 1]],
                                 fus_pool_schd=[(4,2), (4,8)], fus_ch_schd=[32, 32, 16], fus_trgt_grd=(4, 4)),
        }

        def _rep_or_default(x: Optional[List[int]], fallback: List[int]) -> List[int]:
            return fallback if x is None else x

        in_ch_emb  = _rep_or_default(in_ch_emb,  [1] * B)
        in_ch_feat = _rep_or_default(in_ch_feat, [1] * B)

        if dim_emb is None:
            dim_emb = [_per_branch_defaults[b]['dim_emb'] for b in branches]
        if dim_feat is None:
            dim_feat = [_per_branch_defaults[b]['dim_feat'] for b in branches]
        if num_stages is None:
            num_stages = [_per_branch_defaults[b]['num_stages'] for b in branches]
        if stage_depths is None:
            stage_depths = [_per_branch_defaults[b]['stage_depths'] for b in branches]
        if stage_channels is None:
            stage_channels = [_per_branch_defaults[b]['stage_channels'] for b in branches]
        if upsample_emb is None:
            upsample_emb = [_per_branch_defaults[b]['upsample'] for b in branches]
        if fmix_rank_emb is None:
            fmix_rank_emb = [_per_branch_defaults[b]['r_emb'] for b in branches]
        if fmix_rank_feat is None:
            fmix_rank_feat = [_per_branch_defaults[b]['r_feat'] for b in branches]

        def _broadcast_nested(x: Optional[List[List[int | Tuple[int, int]]]],
                              fill: List[int | Tuple[int, int]]) -> List[List[int | Tuple[int, int]]]:
            if x is not None:
                return x
            return [list(fill) for _ in range(B)]

        st_emb  = _broadcast_nested(st_emb,  [1])
        st_feat = _broadcast_nested(st_feat, [1])

        if ks_emb is None:
            ks_emb = [_per_branch_defaults[b]['ks']   for b in branches]
        if ks_feat is None:
            ks_feat = [_per_branch_defaults[b]['ks']   for b in branches]
        if pd_emb is None:
            pd_emb = [_per_branch_defaults[b]['pd']   for b in branches]
        if pd_feat is None:
            pd_feat = [_per_branch_defaults[b]['pd']  for b in branches]
        if dl_emb is None:
            dl_emb = [_per_branch_defaults[b]['dl']   for b in branches]
        if dl_feat is None:
            dl_feat = [_per_branch_defaults[b]['dl']  for b in branches]
        if pool_emb is None:
            pool_emb = [_per_branch_defaults[b]['pool_emb'] for b in branches]
        if pool_feat is None:
            pool_feat = [_per_branch_defaults[b]['pool_feat'] for b in branches]
        if fus_pool_schd is None:
            fus_pool_schd = [_per_branch_defaults[b]['fus_pool_schd'] for b in branches]
        if fus_ch_schd is None:
            fus_ch_schd = [_per_branch_defaults[b]['fus_ch_schd'] for b in branches]
        if fus_trgt_grd is None:
            fus_trgt_grd = [_per_branch_defaults[b]['fus_trgt_grd'] for b in branches]

        def _check_len(name: str, lst: List[Any]) -> None:
            if len(lst) != B:
                raise ValueError(f"'{name}' must have length {B}, got {len(lst)}")

        for name, lst in [
            ('in_ch_emb', in_ch_emb), ('in_ch_feat', in_ch_feat),
            ('dim_emb', dim_emb), ('dim_feat', dim_feat),
            ('num_stages', num_stages), ('stage_depths', stage_depths),
            ('stage_channels', stage_channels), ('upsample_emb', upsample_emb),
            ('ks_emb', ks_emb), ('ks_feat', ks_feat),
            ('st_emb', st_emb), ('st_feat', st_feat),
            ('pd_emb', pd_emb), ('pd_feat', pd_feat),
            ('dl_emb', dl_emb), ('dl_feat', dl_feat),
            ('fmix_rank_emb', fmix_rank_emb), ('fmix_rank_feat', fmix_rank_feat),
            ('pool_emb', pool_emb), ('pool_feat', pool_feat),
            ('fus_pool_schd', fus_pool_schd), ('fus_ch_schd', fus_ch_schd), ('fus_trgt_grd', fus_trgt_grd),
        ]:
            _check_len(name, lst)

        def _normalize_pdrop(x: Optional[float | Dict[str, float]]) -> List[Optional[float]]:
            if isinstance(x, dict):
                return [x.get(b, None) for b in branches]
            return [x] * B  # x may be float or None

        p_drop_mod_s = _normalize_pdrop(p_drop_mod_s)
        p_drop_mod_e = _normalize_pdrop(p_drop_mod_e)

        # ----------------------- build branches -----------------------
        self.branches = nn.ModuleDict()
        for i, bname in enumerate(branches):
            self.branches[bname] = Branch(
                in_ch_emb        = in_ch_emb[i],
                in_ch_feat       = in_ch_feat[i],
                dim_emb          = dim_emb[i],
                dim_feat         = dim_feat[i],
                use_emb          = use_emb,
                use_feat         = use_feat,
                num_stages       = num_stages[i],
                stage_depths     = stage_depths[i],
                stage_channels   = stage_channels[i],
                shared_dim       = shared_dim,
                upsample_emb     = upsample_emb[i],
                fus_pool_schd    = fus_pool_schd[i],
                fus_ch_schd      = fus_ch_schd[i],
                fus_trgt_grd     = fus_trgt_grd[i],
                ks_emb           = ks_emb[i],
                ks_feat          = ks_feat[i],
                st_emb           = st_emb[i],
                st_feat          = st_feat[i],
                pd_emb           = pd_emb[i],
                pd_feat          = pd_feat[i],
                dl_emb           = dl_emb[i],
                dl_feat          = dl_feat[i],
                pool_emb         = pool_emb[i],
                pool_feat        = pool_feat[i],
                p_drop_block     = p_drop_block,
                p_droppath_block = p_droppath_block,
                p_drop_proj      = p_drop_proj,
                fmix_rank_emb    = fmix_rank_emb[i],
                fmix_rank_feat   = fmix_rank_feat[i],
                cross_se         = cross_se,
                p_drop_mod_s     = p_drop_mod_s[i],
                p_drop_mod_e     = p_drop_mod_e[i],
            )

        # ----------------------- multimodal fusion -----------------------
        dims_mod = [fus_ch_schd[i][-1] * int(np.prod(fus_trgt_grd[i])) for i in range(B)]
        dim_min = min(branch_out_ch[-1] for branch_out_ch in stage_channels)
        dim_out = max(dims_mod)
        if B > 1:
            self.mm_fusion = GLRFusion(
                dims_mod     = dims_mod,
                rank_pair    = dim_min,
                alloc_hidden = 2 * dim_min,
                dim_out      = dim_out,
                p_drop_mod_s = p_drop_mod_s,
                p_drop_mod_e = p_drop_mod_e,
            )
        else:
            self.mm_fusion = None

        # ----------------------- classifiers -----------------------
        self.individual_classifier = None
        self.group_classifier = None

        if ind_cls_heads:
            self.individual_classifier = ClassificationHead(
                dim             = dim_out,
                head_names      = list(ind_cls_heads),
                use_prototypes  = use_prototypes,
                trunk_hidden    = trunk_hidden,
                lora_adapter    = lora_adapter,
                adapter_hidden  = adapter_hidden,
                residual        = cls_residual,
                warmup_epochs   = proto_warmup_epochs,
                classifier_type = cls_head_type,
            )

        if grp_cls_heads:
            self.group_classifier = ClassificationHead(
                dim             = dim_out,
                head_names      = list(grp_cls_heads),
                use_prototypes  = use_prototypes,
                trunk_phi_h     = trunk_phi_h,
                trunk_phi       = trunk_phi,
                trunk_hidden    = trunk_hidden,
                adapter_hidden  = dim_out,
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
            z_branch.append(branch_module(x_feat, m_feat, x_emb, m_emb, epoch))

        if len(self.branches) > 1:
            z = self.mm_fusion(z_branch, epoch)
        else:
            z = z_branch[0]

        z_outs: Dict[str, Any] = {}
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


