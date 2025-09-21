import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict

from models.base_model import BaseModel
from models.SyntalNet import Branch
from models.classifier import ClassificationHead


class SingleBranchNet(BaseModel):
    def __init__(
        self,
        *,
        branch_key: str,
        in_ch_emb: int                 = 1,
        in_ch_feat: int                = 1,
        dim_emb: int                   = 1024,
        dim_feat: int                  = 394,
        use_emb: bool                  = True,
        use_feat: bool                 = True,
        num_blocks: int                = 3,
        upsample_emb: bool             = False,
        shared_dim: int                = 128,
        shared_out_ch: int             = 32,
        ch_expansion: List[int]        = [2, 2],
        stride_emb: List[int]          = [(1,2), (1,2), (1,2)],
        branch_residual: bool          = True,
        cross_se: bool                 = True,
        ind_cls_heads: Tuple[str, ...] = (),
        grp_cls_heads: Tuple[str, ...] = (),
        use_prototypes: bool           = True,
        proto_warmup_epochs: int       = 5,
        cls_residual: bool             = True,
        cls_head_type: str             = "cosine",
    ):
        super().__init__()

        self.branch_key = branch_key

        self.branch = Branch(
            in_ch_emb     = in_ch_emb,
            in_ch_feat    = in_ch_feat,
            dim_emb       = dim_emb,
            dim_feat      = dim_feat,
            use_emb       = use_emb,
            use_feat      = use_feat,
            num_stages    = num_blocks,
            upsample_emb  = upsample_emb,
            shared_dim    = shared_dim,
            shared_out_ch = shared_out_ch,
            fusion_ch_expand  = ch_expansion,
            ks_emb        = [(5, 1), (5, 1), (5, 1)],
            ks_feat       = [5, 5, 5],
            st_emb        = stride_emb,
            st_feat       = [1, 1, 1],
            pd_emb        = [(2, 0), (2, 0), (2, 0)],
            pd_feat       = [2, 2, 2],
            dl_emb        = [1, 1, 1],
            dl_feat       = [1, 1, 1],
            residual      = branch_residual,
            cross_se      = cross_se,
        )

        dim_out = 2 * shared_out_ch * ch_expansion[0] * ch_expansion[1]
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

        x_feat, m_feat, x_emb, m_emb = batch_data[self.branch_key]
        z = self.branch(x_feat, m_feat, x_emb, m_emb)

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


