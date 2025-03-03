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
                 num_encoder_layers,
                 num_decoder_layers,
                 dropout=.1,
                 only_encoder=False,
                 device=None,
                 ):
        """
        Args:
            embedding_dim: transformer embedding dim
            nhead: number of attention heads in
            dim_feedforward: feedforward dim in each transformer layer
            num_encoder_layers: number of src encoding layers in transformer
            num_decoder_layers: number of target decoding layers in transformer
            dropout: dropout to use for each layer
            only_encoder: ignore vector encoding, just encode the board
                useful when there is no src vector
            device: device to put stuff on
        """
        super().__init__(device=device)
        # flatten the middle sequence
        self.flat = nn.Flatten(start_dim=1, end_dim=-2)
        self.cls_enc = nn.Embedding(num_embeddings=1,
                                    embedding_dim=embedding_dim,
                                    device=self.device,
                                    )
        self.only_encoder = only_encoder
        if self.only_encoder:
            self.trans = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    device=self.device,
                    batch_first=True,
                ),
                num_layers=num_encoder_layers,
            )
        else:
            self.trans = nn.Transformer(
                d_model=embedding_dim,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                device=self.device,
            )

    def forward_by_layer(self, X, src, perm=None, max_batch=None):
        """
        Args:
            X: board embedding (M, D1, ... DN, E)
            src: src embedding (M, S, E)
                we will never mask the source, as this is not a language model
            perm: permutation of (0,...,N-1), represents order of dimensions that we apply the transformer to
            max_batch: max batch to send through transformer
        """
        # TODO do cls somehow
        shape = X.shape
        N = len(shape) - 2
        if perm is None:
            perm = list(range(N))
        for dim_i in perm:
            dim_i = dim_i + 1  # i is now between 1 and N (inclusive)
            # (M, D1, ... DN, E) -> (M, D{i+1},...,DN, D1, ... Di, E)
            # does this through the permutation (0,i+1,...,N,1,...,i,N+1)
            X = torch.permute(X,
                              [0] +
                              list(range(dim_i + 1, N + 1)) +
                              list(range(1, dim_i + 1)) +
                              [N + 1]
                              )

            temp_shape = X.shape
            X.flatten(0, N)
            # now (M*D1*...*DN,Di,E)

            # TODO: break up the first dim if necessary
            X = self.transforward(X=X, src=src)

            X = X.reshape(temp_shape)
            # shape (M, D{i+1},...,DN, D1, ... Di, E)

            # inverse of the first permute
            # (0,i+1,..., N ,  1  ,...,i,N+1) to
            # (0,1, ... ,N-i,N-i+1,...,N,N+1)
            # standard form: (0,N-i+1,...,N,1,...,N-i,N+1)

            X = torch.permute(X,
                              [0] +
                              list(range(N - dim_i + 1, N + 1)) +
                              list(range(1, N - dim_i + 1)) +
                              [N + 1]
                              )
        return X

    def forward_some_dims(self, X, src, dims, max_batch=None):
        """
        Args:
            X: board embedding (M, D1, ... DN, E)
            src: src embedding (M, S, E)
                we will never mask the source, as this is not a language model
            dims: list of dims to use transformer on, list of indices in [0,N-1]
            max_batch: max batch to send through transformer
        """
        dims = list(dims)
        shape = X.shape
        N = len(shape) - 2
        # sends (0,...,N-1) to (0,...*without dims,N-1, dims)
        perm = [i for i in range(N) if i not in dims] + dims

        inv_perm = sorted(range(N), key=lambda i: perm[i])

        X = torch.permute(X,
                          [0] + [i + 1 for i in perm] + [N + 1],
                          )
        # shape (M,{D1,...D{*without dims},DN, D{dims}},E) (does perm permutation on the middle N dimeensions)

        temp_shape = X.shape

        X = X.flatten(0, N - len(dims))
        # shape (*,D{dims},E)
        X = X.flatten(1, len(dims))
        # shape (*,D{dims_1}*...*D{dims_k},E)

        # TODO: how do we join all the cls encodings
        cls = self.cls_enc(torch.zeros((X.shape[0], 1),
                                       dtype=torch.long,
                                       device=self.device,
                                       ))
        # shape (*,1,E)

        X = self.transforward(X=X, src=src)

        X = X.reshape(temp_shape)  # TODO: suspicious, but probably fine to do this
        X = torch.permute(X,
                          [0] + [i + 1 for i in inv_perm] + [N + 1],
                          )
        return X

    def transforward(self, X, src=None):
        if self.only_encoder:
            X = self.trans.forward(X)
        else:
            X = self.trans.forward(src=src, tgt=X)
        return X

    def forward(self, X, src):
        """
        Args:
            X: board embedding (M, D1, ... DN, E)
            src: src embedding (M, S, E)
                we will never mask the source, as this is not a language model
        Returns:
            (M, D1, ... DN, E), (M, E)
        """
        shape = X.shape

        # (M, D1*...*DN, E)
        X = self.flat(X)

        # (M, 1, E)
        cls = self.cls_enc(torch.zeros((shape[0], 1),
                                       dtype=torch.long,
                                       device=self.device,
                                       ))

        # (M, 1 + D1*...*DN, E)
        X = torch.cat((cls, X), dim=1)
        X = self.transforward(X, src=src)

        # CLS is the 0th elemnet of the sequence, we will separate it
        # then reshape X to its original shape
        return X[:, 1:, :].reshape(shape), X[:, 0, :]


class TransFormerEmbedder(TransFormer):
    """
    ignores src encoding, just encodes board
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
        """
        Args:
            embedding_dim: transformer embedding dim
            nhead: number of attention heads in
            dim_feedforward: feedforward dim in each transformer layer
            num_layers: overall nubmer of layers
            dropout: dropout to use for each layer
            device: device to put stuff on
        """
        super().__init__(embedding_dim=embedding_dim,
                         nhead=nhead,
                         dim_feedforward=dim_feedforward,
                         num_encoder_layers=num_layers,
                         num_decoder_layers=num_layers,  # num_decoders are irrelevant
                         dropout=dropout,
                         only_encoder=True,
                         device=device,
                         )

    def forward(self, X, src=None):
        return super().forward(X=X, src=src)


if __name__ == '__main__':
    import itertools

    # try teaching model to distinguaish its [CLS] embedding from random noise input

    embedding_dim = 16
    out_dim = 1
    trans = TransFormerEmbedder(embedding_dim=embedding_dim,
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
            test_trans_out, cls_trans_out = trans.forward(test)
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
    plt.close()

    embedding_dim = 16
    out_dim = 1
    trans = TransFormer(embedding_dim=embedding_dim,
                        nhead=2,
                        dim_feedforward=128,
                        num_encoder_layers=2,
                        num_decoder_layers=2
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
            testsrc = torch.rand((1, 10, embedding_dim))
            test_trans_out, cls_trans_out = trans.forward(test, testsrc)
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
