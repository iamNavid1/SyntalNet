from .base_branch import SingleBranchNet

class SyntalAudioNet(SingleBranchNet):
    def __init__(self, **kwargs):
        super().__init__(branch_key="acoustic", **kwargs)
