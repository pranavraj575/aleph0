import torch
from torch import nn

from aleph0.networks.architect.middle.former import Former

class TransFormer(Former):
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
    import itertools

    # try teaching model to distinguaish its [CLS] embedding from random noise input

    embedding_dim = 16
    out_dim = 1
    trans = TransFormer(embedding_dim=embedding_dim,
                        nhead=2,
                        dim_feedforward=128,
                        num_layers=2,
                        )
    # since transformers learn embeddings and not values, we will suppliment it with a simple linear map output
    end = nn.Linear(in_features=embedding_dim, out_features=out_dim)
    test_out = None
    cls_out = None
    # too large learning rate will break this easy example
    optim = torch.optim.Adam(itertools.chain(trans.parameters(), end.parameters()), lr=.00005)
    losses = []
    out_values = []
    cls_values = []
    for i in range(2000):
        overall_loss = torch.zeros(1)
        overall_out = torch.zeros(1)
        overall_cls = torch.zeros(1)
        batch_size = 1
        for _ in range(batch_size):
            test = torch.rand((1, 2, 3, 1, 4, embedding_dim))
            test_trans_out, cls_trans_out = trans(test)
            test_out, cls_out = end(test_trans_out), end(cls_trans_out)
            # need to get around torch batch norm
            crit = nn.MSELoss()
            targ = torch.zeros_like(test_out)
            loss = crit(test_out, targ)

            targ2 = torch.ones_like(cls_out)
            crit2 = nn.MSELoss()
            loss2 = crit2(cls_out, targ2)

            overall_loss += (loss + loss2)/2
            overall_cls += torch.mean(cls_out).detach()
            overall_out += torch.mean(test_out).detach()
        overall_loss = overall_loss/batch_size
        overall_loss.backward()
        losses.append(overall_loss.item()/batch_size)
        cls_values.append(overall_cls.item()/batch_size)
        out_values.append(overall_out.item()/batch_size)
        print(i, end='\r')
        optim.step()
    from matplotlib import pyplot as plt

    plt.plot(losses)
    plt.title('losses')
    plt.show()
    plt.plot(out_values, label='out values', color='purple')
    plt.plot([0 for _ in range(len(out_values))], '--', label='out target', color='purple')
    plt.plot(cls_values, label='cls values', color='red')
    plt.plot([1 for _ in range(len(out_values))], '--', label='cls target', color='red')
    plt.legend()
    plt.show()
