# former superclass, goes from embedding to [cls output], embedding
# [cls output] is a special embedding vector that is used for maps from the whole board
# an example would be a value extimate of the whole board, which would use [cls output] as input
from torch import nn


class Former(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, X, src):
        """
        Args:
            X: board embedding (M, D1, ... DN, E)
            src: additional input vector of arbitrary form, must be handled by subclass
        Returns:
            (M, D1, ... DN, E), (M, E)
            transformed embedding, special [cls output] value
        """
        raise NotImplementedError
