from .SyntalNet import SyntalNet


class SyntalAudioLinguoNet(SyntalNet):
    """Two-branch SyntalNet with acoustic and dialogue branches."""

    def __init__(self, *, branches=None, **kwargs):
        branch_list = branches or ["Acoustic", "Dialogue"]
        super().__init__(branches=branch_list, **kwargs)
