from torch import nn

from aleph0.networks.architect.middle.former import Former


class ChainFormer(Former):
    """
    chains multiple formers together, uses cls output of the last one
    """

    def __init__(self, former_list, device=None):
        super().__init__(device=device)
        self.formers = nn.ModuleList(former_list)

    def forward(self, X, src):
        cls_embedding = None
        for former in self.formers:
            X, cls_embedding = former.forward(X, src)
        return X, cls_embedding
