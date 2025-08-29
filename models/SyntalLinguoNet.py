from .base_branch import SingleBranchNet

class SyntalLinguoNet(SingleBranchNet):
    def __init__(self, **kwargs):
        super().__init__(
            branch_key   = "dialogue", 
            in_ch_emb    = 1,
            in_ch_feat   = 1,
            dim_emb      = 1024,
            dim_feat     = 12,
            num_blocks   = 3,
            upsample_emb = False,
            stride_emb   = [(1,2), (1,2), (1,2)],
            **kwargs,
        )
