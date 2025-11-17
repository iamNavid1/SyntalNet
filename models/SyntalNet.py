import copy
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from typing import List, Tuple, Dict, Optional, Any, Type

import models.builders as build
from models.base_model import BaseModel
from models.utils import ChannelLayerNorm2d
from models.classifier import ClassificationHead
from models.encoder import CNXv2Block, StageTransition
from models.partial import PartialAvgPool2d, PartialGeM
from models.multiperson_fusion import SoSE_X


# -----------------------------------------------------------------------------
#                                Specifications
# -----------------------------------------------------------------------------

VALID_MODALITIES = [
    "face", "pose", "video",
    "turns", "utterance",
    "sentiment", "prosody", "audio",
]

VALID_BRANCHES = ["Videokinetic", "Dialogue", "Acoustic"]

BRANCH_MODALITIES: Dict[str, List[str]] = {
    "Videokinetic": ["video", "face", "pose"],
    "Dialogue": ["utterance", "turns"],
    "Acoustic": ["audio", "prosody", "sentiment"],
}

MODALITY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'video': dict(
        in_ch=1024, stage_chs=[96], num_stages=3, stage_depth=[1,1,2],
        ks=[[(5,1)], [(7,1)], [(7,1)]], pd=[[(2,0)], [(3,0)], [(3,0)]],
        st=[[1]], dl=[[1]], interp=True, pool=[None],
    ),
    'face': dict(
        in_ch=212, stage_chs=[80], num_stages=3, stage_depth=[1,1,2],
        ks=[[(5,1)], [(7,1)], [(7,1)]], pd=[[(2,0)], [(3,0)], [(3,0)]],
        st=[[1]], dl=[[1]], interp=False, pool=[None],
    ),
    'pose': dict(
        in_ch=182, stage_chs=[80], num_stages=3, stage_depth=[1,1,2],
        ks=[[(5,1)], [(7,1)], [(7,1)]], pd=[[(2,0)], [(3,0)], [(3,0)]],
        st=[[1]], dl=[[1]], interp=False, pool=[None],
    ),
    'utterance': dict(
        in_ch=1024, stage_chs=[96], num_stages=3, stage_depth=[1,1,2],
        ks=[[(3,1)], [(3,1)], [(5,1)]], pd=[[(1,0)], [(1,0)], [(2,0)]],
        st=[[1]], dl=[[1]], interp=False, pool=[None],
    ),
    'turns': dict(
        in_ch=12, stage_chs=[80], num_stages=3, stage_depth=[1],
        ks=[[(3,1)], [(3,1)], [(5,1)]], pd=[[(1,0)], [(1,0)], [(2,0)]],
        st=[[1]], dl=[[1]], interp=False, pool=[None],
    ),
    'audio': dict(
        in_ch=512, stage_chs=[96], num_stages=3, stage_depth=[1,1,2],
        ks=[[(5,1)], [(7,1)], [(7,1)]], pd=[[(2,0)], [(3,0)], [(3,0)]],
        st=[[1]], dl=[[1]], interp=False, pool=[None],
    ),
    'sentiment': dict(
        in_ch=5, stage_chs=[48], num_stages=3, stage_depth=[1],
        ks=[[(5,1)], [(7,1)], [(7,1)]], pd=[[(2,0)], [(3,0)], [(3,0)]],
        st=[[1]], dl=[[1]], interp=False, pool=[None],
    ),
    'prosody': dict(
        in_ch=7, stage_chs=[48], num_stages=3, stage_depth=[1],
        ks=[[(5,1)], [(7,1)], [(7,1)]], pd=[[(2,0)], [(3,0)], [(3,0)]],
        st=[[1]], dl=[[1]], interp=False, pool=[None],
    ),
}

# -----------------------------------------------------------------------------
#                                Encoder Blocks
# -----------------------------------------------------------------------------
class EncoderStage(nn.Module):
    """
    One stage with:
      - A list of depth CNXv2Blocks
      - Optional PartialGeM over T
      - Low-rank Feature Mixer over F w/ w/o dimension reduction
    """
    def __init__(
        self,
        C: int,
        depth: int,
        ks: List[int | Tuple[int, int]],
        pd: List[int | Tuple[int, int]],
        st: List[int | Tuple[int, int]],
        dl: List[int | Tuple[int, int]],
        expansion: int | float,
        last_stage: bool,
        act: nn.Module,
        p_drop: float,
        p_droppath: List[float],
        pool: Optional[int | Tuple[int, int]],
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

        if pool is not None:
            p_t, _ = _pair(pool)
        else:
            p_t = None

        self.temporal_pool = None
        if p_t is not None and p_t > 1:
            self.temporal_pool = PartialAvgPool2d((p_t, 1), (p_t, 1))

    def forward(self, x: torch.Tensor, m:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x, m = blk(x, m)
        if self.temporal_pool is not None:
            x, m = self.temporal_pool(x, m)
        return x, m

# -----------------------------------------------------------------------------
#                              Encoder Backbone
# -----------------------------------------------------------------------------
class EncoderBackbone(nn.Module):
    """
    Multi-stage encoder:
      - optional stem StageTransition to first stage width
      - per stage: EncoderStage -> optional SoSE-X -> optional StageTransition to next stage
    """
    def __init__(
        self,
        in_channels: int,
        num_stages: int,
        stage_depths:    List[int],
        stage_channels:  List[int],
        stage_kernels:   List[List[int | Tuple[int, int]]],
        stage_pads:      List[List[int | Tuple[int, int]]],
        stage_strides:   List[List[int | Tuple[int, int]]],
        stage_dilations: List[List[int | Tuple[int, int]]],
        stage_pool:      List[Optional[int | Tuple[int, int]]],
        expansion:       int | float = 1.5,
        act:             nn.Module = nn.GELU,
        p_drop:          float = .05,
        p_droppath:      float = 0.,  # max DropPath
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

        first_C = stage_channels[0]
        self.stem = StageTransition(in_channels, first_C) \
            if in_channels != first_C \
            else None

        total_blocks = sum(stage_depths)
        dp_rates = torch.linspace(0.0, p_droppath, total_blocks).tolist()

        self.stages = nn.ModuleList()
        self.mp_fusion  = nn.ModuleList()
        self.trans  = nn.ModuleList()

        cur = 0
        for i in range(num_stages):
            depth_i = stage_depths[i]
            dp_i = dp_rates[cur : cur+depth_i]
            cur += depth_i

            stage_i = EncoderStage(
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
            )
            self.stages.append(stage_i)

            if apply_mp_fusion and i == num_stages-1:
                self.mp_fusion.append(SoSE_X(num_channels=stage_channels[i], dropout_p=p_drop))
            else:
                self.mp_fusion.append(nn.Identity())

            if i < num_stages - 1:
                if stage_channels[i] != stage_channels[i + 1]:
                    self.trans.append(StageTransition(stage_channels[i], stage_channels[i + 1], norm=True))
                else:
                    self.trans.append(nn.Identity())
            else:
                self.trans.append(nn.Identity())


    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 5 and m.ndim == 5, "x and m must be 5D: (B,P,C,T,F) and/or (B,P,1,T,F)"

        B, P, C, T, F = x.shape
        x = x.view(B * P, C, T, F)
        m = m.reshape(B * P, m.shape[-3], m.shape[-2], m.shape[-1])
        if m.shape[1] != 1:
            m = (m.sum(dim=1, keepdim=True) > 0).to(dtype=x.dtype)
        else:
            m = m.to(dtype=x.dtype)

        if self.stem is not None:
            x, m = self.stem(x, m)

        for stage, fusion, transition in zip(self.stages, self.mp_fusion, self.trans):
            x, m = stage(x, m)
            if not isinstance(fusion, nn.Identity):
                x = fusion(x, m)
            if not isinstance(transition, nn.Identity):
                x, m = transition(x, m)

        x = x.view(B, P, x.shape[-3], x.shape[-2], x.shape[-1])
        m = m.view(B, P, m.shape[-3], m.shape[-2], m.shape[-1])

        return x, m

# -----------------------------------------------------------------------------
#                                   Branch
# -----------------------------------------------------------------------------
class Branch(nn.Module):
    def __init__(
        self,
        # per-modality configs
        mod_in_ch:      Dict[str, int],
        mod_stage_chs:  Dict[str, List[int]],
        num_stages:     Dict[str, int],
        stage_depths:   Dict[str, int],
        ks:             Dict[str, List[List[int | Tuple[int, int]]]],
        st:             Dict[str, List[List[int | Tuple[int, int]]]],
        pd:             Dict[str, List[List[int | Tuple[int, int]]]],
        dl:             Dict[str, List[List[int | Tuple[int, int]]]],
        pool:           Dict[str, List[int | Tuple[int, int] | None]],
        interp:         Optional[Dict[str, bool]] = None,
        # blocks
        p_drop_block: float = 0.05,
        p_droppath_block: float = 0.0,
        activation_block: Type[nn.Module] = nn.GELU,
        # fusion
        mc_fusion_type: str = "bscx",
        mp_fusion: bool = True,
        shared_dim: int = 128,
        # p_drop_proj: float = 0.1,
        # activation_proj: Type[nn.Module] = nn.GELU,
        # modality dropout
    ):
        super().__init__()

        self.mods = list(mod_in_ch.keys())
        self.interp = interp or {}

        self.encoders = nn.ModuleDict()
        self.upsamplers = nn.ModuleDict()

        # build per-modality backbones & projectors
        for m in self.mods:
            enc = EncoderBackbone(
                in_channels    = mod_in_ch[m],
                num_stages     = num_stages[m],
                stage_depths   = stage_depths[m],
                stage_channels = mod_stage_chs[m],
                stage_kernels  = ks[m],
                stage_pads     = pd[m],
                stage_strides  = st[m],
                stage_dilations= dl[m],
                stage_pool     = pool[m],
                act            = activation_block,
                p_drop         = p_drop_block,
                p_droppath     = p_droppath_block,
                apply_mp_fusion= mp_fusion,
            )
            self.encoders[m] = enc

            if self.interp.get(m, False):
                TARGET_LENGTH = 190
                self.upsamplers[m] = nn.Upsample(size=TARGET_LENGTH, mode='linear')

        # intra-modality fusion
        self.mods_order = list(self.mods)
        Cins = [mod_stage_chs[m][-1] for m in self.mods_order]
        if len(self.mods) > 1:
            self.mc_fusion = build.multichannel_fusion(
                variant    = mc_fusion_type,
                Cin_list   = Cins,
                C          = shared_dim,
                out_dim    = shared_dim,
            )
            self.single_pool = None
        else:
            C0 = Cins[0]
            self.mc_fusion = None
            self.single_pool = nn.ModuleDict({
                "gem": PartialGeM(),
                "head": nn.Sequential(
                    nn.LayerNorm(2*C0, eps=1e-5),
                    nn.Linear(2*C0, shared_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                )
            })

    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor],
        m_dict: Dict[str, torch.Tensor],
        epoch: Optional[int] = None
        ) -> torch.Tensor:
        
        any_x = next(iter(x_dict.values()))
        B, P = any_x.shape[:2]

        Zs, Ms = [], []
        for m in self.mods_order:
            x = x_dict[m]  # (B,P,T,F)
            mk = m_dict[m]

            # optional temporal upsample
            if m in self.upsamplers:
                # x: (B,P,T,F) -> (N=BP, C=F, L=T)
                N = B * P
                # move channels to dim-1 and flatten B,P
                xN = x.permute(0, 1, 3, 2).reshape(N, x.shape[-1], x.shape[-2])  # (N, F, T)
                xN = self.upsamplers[m](xN)                                      # (N, F, T')

                x = xN.permute(0, 2, 1).reshape(B, P, xN.shape[-1], xN.shape[-2])  # (B,P,T',F)

                mN = mk.permute(0, 1, 3, 2).reshape(N, mk.shape[-1], mk.shape[-2]).to(xN.dtype)  # (N, F, T)
                mN = torch.nn.functional.interpolate(mN, size=xN.shape[-1], mode='nearest')      # (N, F, T')
                mk = mN.permute(0, 2, 1).reshape(B, P, mN.shape[-1], mN.shape[-2])               # (B,P,T',F)

            # encoder expects (B,P,F,T,1)
            x = x.permute(0, 1, 3, 2).unsqueeze(-1)  # (B,P,F,T',1)
            mk = mk.permute(0, 1, 3, 2).unsqueeze(-1)  # (B,P,F,T',1)
            ze, me = self.encoders[m](x, mk)
            # flatten persons for fusion block
            ze = ze.view(B*P, ze.shape[-3], ze.shape[-2], ze.shape[-1])  # (N,F,T',1)
            me = me.view(B*P, me.shape[-3], me.shape[-2], me.shape[-1])  # (N,F,T',1)
            # expand mask across channels
            me = me.expand(-1, ze.shape[1], -1, -1).to(dtype=ze.dtype)
            Zs.append(ze)
            Ms.append(me)

        if self.mc_fusion is not None:
            z_branch = self.mc_fusion(Zs, Ms)                           # (N, shared_dim)
        else:
            # single modality path (match BSC_X head behavior)
            Z, M = Zs[0], Ms[0]
            mean = (Z*M).sum(dim=(2,3)) / M.sum(dim=(2,3)).clamp_min(1e-6)
            gem  = self.single_pool["gem"](Z, M)
            vec  = torch.cat([mean, gem], dim=-1)
            z_branch = self.single_pool["head"](vec)

        return z_branch  # shape (B*P, shared_dim)

# -----------------------------------------------------------------------------
#                                  SyntalNet
# -----------------------------------------------------------------------------
class SyntalNet(BaseModel):
    def __init__(
        self,
        modalities: Optional[List[str]] = None,
        branches: Optional[List[str]] = None,
        shared_dim: int = 128,
        # in-branch fusion
        mc_fusion_type: str = "bscx",
        # cross-branch fusion
        mm_fusion_type: str = "glrx",
        rank_pair: int = 12,
        alloc_hidden: int = 24,
        # encoder / stage hyperparams
        p_drop_block: float = 0.12,
        p_droppath_block: float = 0.0,
        mp_fusion: bool = True,
        # classification
        ind_cls_heads: Tuple[str, ...] = (),
        grp_cls_heads: Tuple[str, ...] = (),
        cls_head_type: str = "cosine",
        use_prototypes: bool = True,
        trunk_phi_h: Optional[int] = 96,
        trunk_phi: Optional[int] = 96,
        trunk_hidden: Optional[int] = 96,
        lora_adapter: bool = True,
        adapter_hidden: Optional[int] = 24,
        proto_warmup_epochs: int = 5,
        cls_residual: bool = True,
    ):
        super().__init__()

        # ----------------------- defaults -----------------------
        modalities = modalities or list(VALID_MODALITIES)
        branches = branches or list(VALID_BRANCHES)

        if not set(modalities).issubset(VALID_MODALITIES):
            raise ValueError("modalities must be drawn from {'face', 'pose', 'video', 'turns', 'utterance', 'sentiment', 'prosody', 'audio'}")
        if not set(branches).issubset(VALID_BRANCHES):
            raise ValueError("branches must be drawn from {'Videokinetic','Dialogue','Acoustic'}")

        branch_modalities = {
            name: [m for m in BRANCH_MODALITIES[name] if m in modalities]
            for name in BRANCH_MODALITIES
        }

        _per_modality_defaults = {
            k: copy.deepcopy(MODALITY_DEFAULTS[k])
            for k in MODALITY_DEFAULTS
        }

        # ----------------------- build branches -----------------------
        self.branches = nn.ModuleDict()

        for bname in branches:
            mods_b = branch_modalities[bname]
            if not mods_b:
                continue
            
            mod_in_ch        = {m: _per_modality_defaults[m]['in_ch'] for m in mods_b}
            mod_stage_chs    = {m: _per_modality_defaults[m]['stage_chs'] for m in mods_b}
            mod_num_stages   = {m: _per_modality_defaults[m]['num_stages'] for m in mods_b}
            mod_stage_depths = {m: _per_modality_defaults[m]['stage_depth'] for m in mods_b}
            ks               = {m: _per_modality_defaults[m]['ks'] for m in mods_b}
            pd               = {m: _per_modality_defaults[m]['pd'] for m in mods_b}
            st               = {m: _per_modality_defaults[m]['st'] for m in mods_b}
            dl               = {m: _per_modality_defaults[m]['dl'] for m in mods_b}
            pool             = {m: _per_modality_defaults[m]['pool'] for m in mods_b}
            interp           = {m: _per_modality_defaults[m]['interp'] for m in mods_b}

            self.branches[bname] = Branch(
                mod_in_ch=mod_in_ch, mod_stage_chs=mod_stage_chs,
                num_stages=mod_num_stages, stage_depths=mod_stage_depths,
                ks=ks, st=st, pd=pd, dl=dl, pool=pool, interp=interp,
                p_drop_block=p_drop_block, p_droppath_block=p_droppath_block,
                mc_fusion_type=mc_fusion_type, mp_fusion=mp_fusion, shared_dim=shared_dim,
            )
 
        # ----------------------- multimodal fusion -----------------------
        M = len(self.branches)
        if M > 1:
            self.mm_fusion = build.multimodal_fusion(
                variant    = mm_fusion_type,
                num_mod    = M,
                dims_mod   = shared_dim,
                rank_pair  = rank_pair,
                alloc_hidden = alloc_hidden,
            )
        else:
            self.mm_fusion = None

        # ----------------------- classifiers -----------------------
        self.individual_classifier = None
        self.group_classifier = None

        if ind_cls_heads:
            self.individual_classifier = ClassificationHead(
                dim             = shared_dim,
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
                dim             = shared_dim,
                head_names      = list(grp_cls_heads),
                use_prototypes  = use_prototypes,
                trunk_phi_h     = trunk_phi_h,
                trunk_phi       = trunk_phi,
                trunk_hidden    = trunk_hidden,
                adapter_hidden  = adapter_hidden,
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
        for bname, branch in self.branches.items():
            x_dict = {m: batch_data[bname.lower()][m][0] for m in branch.mods}
            m_dict = {m: batch_data[bname.lower()][m][1] for m in branch.mods}
            z_b = branch(x_dict, m_dict, epoch)  # dict m -> (N,C_last)
            z_branch.append(z_b)

        if len(self.branches) > 1:
            z = self.mm_fusion(z_branch)
        else:
            z = z_branch[0]

        z_outs: Dict[str, Any] = {"backbone": z}
        logits: Dict[str, Dict[str, torch.Tensor]] = {}
        
        if self.individual_classifier:
            ind_features, ind_logits = self.individual_classifier(z, epoch)
            z_outs.update({'individual': ind_features})
            logits.update({'individual': ind_logits})
            
        if self.group_classifier:
            grp_features, grp_logits = self.group_classifier(z, epoch)
            z_outs.update({'group': grp_features})
            logits.update({'group': grp_logits})

        return z_outs, logits


