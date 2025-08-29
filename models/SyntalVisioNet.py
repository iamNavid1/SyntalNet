from .base_branch import SingleBranchNet

class SyntalVisioNet(SingleBranchNet):
    def __init__(self, **kwargs):
        super().__init__(
            branch_key   = "videokinetic", 
            in_ch_emb    = 1,
            in_ch_feat   = 1,
            dim_emb      = 1024,
            dim_feat     = 394,
            num_blocks   = 3,
            upsample_emb = True,
            stride_emb   = [(1,2), (1,2), (1,2)],
            **kwargs,
        )
