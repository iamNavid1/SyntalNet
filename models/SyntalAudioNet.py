from .base_branch import SingleBranchNet

class SyntalAudioNet(SingleBranchNet):
    def __init__(self, **kwargs):
        super().__init__(
            branch_key   = "acoustic", 
            in_ch_emb    = 1,
            in_ch_feat   = 1,
            dim_emb      = 512,
            dim_feat     = 12,
            num_blocks   = 3,
            upsample_emb = False,
            stride_emb   = [(1,1), (1,2), (1,2)],
            **kwargs,
        )
