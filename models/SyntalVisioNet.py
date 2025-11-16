from .base_branch import SingleBranchNet

class SyntalVisioNet(SingleBranchNet):
    def __init__(self, **kwargs):
        super().__init__(branch_key="videokinetic", **kwargs)
