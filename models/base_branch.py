import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Type, Any

from models.base_model import BaseModel
from models.SyntalNet import Branch, BRANCH_MODALITIES, MODALITY_DEFAULTS, VALID_BRANCHES
from models.classifier import ClassificationHead


class SingleBranchNet(BaseModel):
    def __init__(
        self,
        *,
        branch_key: str,
        modalities: Optional[List[str]] = None,
        modality_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        shared_dim: int = 128,
        cross_se: bool = True,
        p_drop_block: float = 0.12,
        p_droppath_block: float = 0.0,
        activation_block: Type[nn.Module] = nn.GELU,
        ind_cls_heads: Tuple[str, ...] = (),
        grp_cls_heads: Tuple[str, ...] = (),
        cls_head_type: str = "cosine",
        use_prototypes: bool = True,
        proto_warmup_epochs: int = 5,
        cls_residual: bool = True,
        trunk_phi_h: Optional[int] = 96,
        trunk_phi: Optional[int] = 96,
        trunk_hidden: Optional[int] = 96,
        lora_adapter: bool = True,
        adapter_hidden: Optional[int] = 24,
    ):
        super().__init__()

        self.branch_key = branch_key.lower()
        self.branch_name = self.branch_key.capitalize()

        if self.branch_name not in BRANCH_MODALITIES:
            raise ValueError(
                f"branch_key must be one of {sorted(k.lower() for k in VALID_BRANCHES)}, got '{branch_key}'"
            )

        available_modalities = BRANCH_MODALITIES[self.branch_name]
        if modalities is None:
            selected_modalities = list(available_modalities)
        else:
            invalid = set(modalities) - set(available_modalities)
            if invalid:
                raise ValueError(
                    f"Modalities {sorted(invalid)} are not supported for branch '{self.branch_name}'"
                )
            if not modalities:
                raise ValueError("At least one modality must be provided for the branch")
            selected_modalities = list(modalities)

        self.selected_modalities = selected_modalities
        modality_overrides = modality_overrides or {}

        mod_in_ch: Dict[str, int] = {}
        mod_stage_chs: Dict[str, List[int]] = {}
        mod_num_stages: Dict[str, int] = {}
        mod_stage_depths: Dict[str, List[int]] = {}
        ks: Dict[str, List[List[int | Tuple[int, int]]]] = {}
        st: Dict[str, List[List[int | Tuple[int, int]]]] = {}
        pd: Dict[str, List[List[int | Tuple[int, int]]]] = {}
        dl: Dict[str, List[List[int | Tuple[int, int]]]] = {}
        pool: Dict[str, List[Optional[int | Tuple[int, int]]]] = {}
        interp: Dict[str, bool] = {}

        for mod in selected_modalities:
            if mod not in MODALITY_DEFAULTS:
                raise ValueError(f"Unknown modality '{mod}' for branch '{self.branch_name}'")
            cfg = copy.deepcopy(MODALITY_DEFAULTS[mod])
            cfg.update(modality_overrides.get(mod, {}))

            mod_in_ch[mod] = cfg["in_ch"]
            mod_stage_chs[mod] = cfg["stage_chs"]
            mod_num_stages[mod] = cfg["num_stages"]
            mod_stage_depths[mod] = cfg["stage_depth"]
            ks[mod] = cfg["ks"]
            st[mod] = cfg["st"]
            pd[mod] = cfg["pd"]
            dl[mod] = cfg["dl"]
            pool[mod] = cfg["pool"]
            interp[mod] = cfg["interp"]

        self.branches = nn.ModuleDict()
        self.branches[self.branch_name] = Branch(
            mod_in_ch=mod_in_ch,
            mod_stage_chs=mod_stage_chs,
            num_stages=mod_num_stages,
            stage_depths=mod_stage_depths,
            ks=ks,
            st=st,
            pd=pd,
            dl=dl,
            pool=pool,
            interp=interp,
            p_drop_block=p_drop_block,
            p_droppath_block=p_droppath_block,
            activation_block=activation_block,
            cross_se=cross_se,
            shared_dim=shared_dim,
        )

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

    def forward(self, batch_data, epoch: Optional[int] = None):
        if self.branch_key not in batch_data:
            raise KeyError(f"Batch is missing branch '{self.branch_key}'")

        branch_data = batch_data[self.branch_key]

        x_dict = {}
        m_dict = {}
        for mod in self.selected_modalities:
            if mod not in branch_data:
                raise KeyError(f"Batch branch '{self.branch_key}' is missing modality '{mod}'")
            x_dict[mod], m_dict[mod] = branch_data[mod]

        z = self.branches[self.branch_name](x_dict, m_dict, epoch)

        z_outs: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {}
        logits: Dict[str, Dict[str, torch.Tensor]] = {}
        z_outs['backbone'] = z
        if self.individual_classifier:
            ind_features, ind_logits = self.individual_classifier(z, epoch)
            z_outs["individual"] = ind_features
            logits["individual"] = ind_logits
        if self.group_classifier:
            grp_features, grp_logits = self.group_classifier(z, epoch)
            z_outs["group"] = grp_features
            logits["group"] = grp_logits

        return z_outs, logits


