from .base_branch import SingleBranchNet

class SyntalLinguoNet(SingleBranchNet):
    def __init__(self, **kwargs):
        super().__init__(branch_key="dialogue", **kwargs)
