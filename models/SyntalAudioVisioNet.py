from .SyntalNet import SyntalNet


class SyntalAudioVisioNet(SyntalNet):
    """Two-branch SyntalNet with acoustic and videokinetic branches."""

    def __init__(self, *, branches=None, **kwargs):
        branch_list = branches or ["Videokinetic", "Acoustic"]
        super().__init__(branches=branch_list, **kwargs)
