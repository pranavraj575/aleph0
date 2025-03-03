"""
to collapse a sequence3
"""
import torch
from torch import nn
from aleph0.networks.ffn import FFN


class Collapse(nn.Module):
    """
        collapses a sequence down to a single vector by doing a weighted sum of mapped values
    """

    def forward(self, X):
        """
        :param X: (batch size, *, embedding_dim)
        :return: (batch size, embedding_dim)
            for each element of the batch, should be a weighted average of all embedding values
        """
        raise NotImplementedError


class CollapseFFN(Collapse):
    """
    collapses a sequence down to a single vector by doing a weighted sum of mapped values

    learns a FFN to determine relevance of each element, as well as values for each
    similar to a multihead attention, with a query of 1 element
    """

    def __init__(self, embedding_dim: int, hidden_key_layers=None, hidden_value_layers=None):
        """
        if hideen layers is none, the FFN learned is a simple linear map
        """
        super().__init__()

        # neural net that ends at a scalar
        self.ffn_key = FFN(input_dim=embedding_dim,
                           output_dim=1,
                           hidden_layers=hidden_key_layers,
                           )
        self.ffn_value = FFN(input_dim=embedding_dim,
                             output_dim=embedding_dim,
                             hidden_layers=hidden_value_layers,
                             )
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(-1)

    def forward(self, X):
        """
        :param X: (batch size, *, embedding_dim)
        :return: (batch size, embedding_dim)
            for each element of the batch, should be a weighted average of all embedding values
        """
        batch_size = X.shape[0]
        relevances = self.ffn_key(X)
        # relevances is (batch size, *, 1)

        # (batch size, M), where M is the product of all the dimensions in *
        relevances = relevances.view((batch_size, -1))
        weights = self.softmax(relevances)

        # (batch size, 1, M)
        weights = weights.unsqueeze(1)

        # values mapping does not change embedding_dim, still (batch size, M, embedding_dim)
        values = self.ffn_value.forward(
            X.view((batch_size, -1, self.embedding_dim))  # (batch size, M, embedding_dim)
        )

        # (batch size, 1, embedding_dim)
        output = torch.bmm(weights, values)

        return output.view((batch_size, self.embedding_dim))


class CollapseMHA(Collapse):
    """
    collapses a squence to a single vector using multihead attention, with a query of 1 element
    """

    def __init__(self, embedding_dim: int, num_heads=1, dropout=.1, device=None):
        """
        Args:
            embedding_dim: embedding dimension of input, and of resulting output
            hidden_key_layers: hidden layers to create keys for each embedding, if None, uses a linear map
            hidden_value_layers: hidden layers to create values for each embedding, if None, uses a linear map
        """
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        self.query_embedding = nn.Embedding(num_embeddings=1,
                                            embedding_dim=embedding_dim,
                                            device=self.device,
                                            )  # the query
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         batch_first=True,
                                         device=self.device,
                                         )

    def forward(self, X):
        """
        :param X: (batch size, *, embedding_dim)
        :return: (batch size, embedding_dim)
        """
        batch_size = X.shape[0]
        X = X.view((batch_size, -1, self.embedding_dim))  # (batch size, M, embedding_dim), where M is the product of *
        output = self.mha.forward(
            query=self.query_embedding.forward(torch.zeros((batch_size, 1), device=self.device, dtype=torch.long)),
            # (batch_size, 1, embedding_dim)
            key=X,  # (batch size, M, embedding_dim)
            value=X,  # (batch size, M, embedding_dim)
        )[0]
        # shaped (batch_size,1,embedding_dim)

        return output.view((batch_size, -1))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    embed_dim = 64
    trial = 0
    seq_len = 69

    for cc in (CollapseMHA(embedding_dim=embed_dim, num_heads=1, dropout=0.),  # this one is best, maybe bc task is ez
               CollapseMHA(embedding_dim=embed_dim, num_heads=2, dropout=0.),
               CollapseMHA(embedding_dim=embed_dim, num_heads=4, dropout=0.),
               CollapseFFN(embedding_dim=embed_dim, hidden_key_layers=[32], hidden_value_layers=[32]),  # best
               CollapseFFN(embedding_dim=embed_dim, hidden_key_layers=[16], hidden_value_layers=[16]),
               CollapseFFN(embedding_dim=embed_dim, hidden_key_layers=[], hidden_value_layers=[]),
               ):
        X = torch.rand((1, seq_len, embed_dim))*torch.arange(seq_len).view((1, seq_len, 1))
        collapsed = cc.forward(X=X)
        print(X.shape, collapsed.shape)

        optim = torch.optim.Adam(cc.parameters())
        critera = nn.MSELoss()
        losses = []
        for i in range(1000):
            optim.zero_grad()
            X = torch.rand((64, seq_len, embed_dim))*torch.arange(seq_len).view((1, seq_len, 1))
            Y = X[:, 0]  # (_,embed_dim), the 'correct' embeddings
            loss = critera.forward(input=cc.forward(X), target=Y)
            loss.backward()
            optim.step()
            losses.append(torch.log(loss).item())
            print(loss, end='           \r')
        plt.plot(losses, label='trial ' + str(trial))
        trial += 1
    plt.legend()
    plt.ylabel('log loss')
    plt.show()
