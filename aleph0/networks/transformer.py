import torch
from torch import nn


class TransArchitect(nn.Module):
    """
    takse a board embedding (M, D1, ... DN, E) and collapses into a sequence
            (M, D1*...*DN, E)
        then adds a [CLS] token
            (M, 1 + D1*...*DN, E)
        then does trans former things and uncollapses the sequence and the encoded [CLS] token
            (M, D1, ... DN, E), (M, E)
        returns this shape
    """

    def __init__(self,
                 embedding_dim,
                 nhead,
                 dim_feedforward,
                 num_layers,
                 dropout=.1,
                 device=None,
                 ):
        super().__init__()
        # flatten the middle sequence
        self.flat = nn.Flatten(start_dim=1, end_dim=-2)
        self.cls_enc = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)
        self.trans = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                device=device,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, X):
        """
        Args:
            X: board embedding (M, D1, ... DN, E)
        Returns:
            (M, D1, ... DN, E), (M, E)
        """
        shape = X.shape

        # (M, D1*...*DN, E)
        X = self.flat(X)

        # (M, 1, E)
        cls = self.cls_enc(torch.zeros((shape[0], 1), dtype=torch.long))

        # (M, 1 + D1*...*DN, E)
        X = torch.cat((cls, X), dim=1)
        X = self.trans.forward(X)

        # CLS is the 0th elemnet of the sequence, we will separate it
        # then reshape X to its original shape
        return X[:, 1:, :].reshape(shape), X[:, 0, :]


if __name__ == '__main__':
    embedding_dim = 16
    trans = TransArchitect(embedding_dim=embedding_dim,
                           nhead=2,
                           dim_feedforward=69,
                           num_layers=2,
                           )
    test = torch.rand((1, 2, 2, 3, 4, embedding_dim))
    test_out,cls_out=trans(test)
    print(test_out.shape)
    print(cls_out.shape)
