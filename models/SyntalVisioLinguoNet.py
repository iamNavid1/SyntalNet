from .SyntalNet import SyntalNet


class SyntalVisioLinguoNet(SyntalNet):
    """Two-branch SyntalNet with videokinetic and dialogue branches."""

    def __init__(self, *, branches=None, **kwargs):
        branch_list = branches or ["Videokinetic", "Dialogue"]
        super().__init__(branches=branch_list, **kwargs)
