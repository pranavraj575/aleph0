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

    def forward_by_layer(self, X, src, collapse_fn, perm=None, max_batch=None):
        """
        Args:
            X: board embedding (M, D1, ... DN, E)
            src: src embedding (M, S, E)
                we will never mask the source, as this is not a language model
            collapse_fn: collapses a sequence (M,*,E) to (M,E), usually using netowrks from self
            perm: permutation of (0,...,N-1), represents order of dimensions that we apply the transformer to
            max_batch: max batch to send through transformer
        """
        shape = X.shape
        N = len(shape) - 2
        if perm is None:
            perm = list(range(N))
        clses = []
        for dim_i in perm:
            X, cls_i = self.forward_some_dims(X=X, src=src, dims=[dim_i], collapse_fn=collapse_fn, max_batch=max_batch)
            # x does not change shape, cls_i is (M,E), viewed as (M,1,E)
            clses.append(cls_i.view(shape[0], 1, shape[-1]))
        # apply collapse once more to all the cls_i, which concatenated are (M,D,E)
        cls = collapse_fn(torch.concatenate(clses, dim=1))
        return X, cls

    def forward_some_dims(self, X, src, dims, collapse_fn, max_batch=None):
        """
        uses the transformer only on some dimensions
            this greatly reduces the size of sequences, at the cost of elements only being able to pay attention
                to elemnts that only vary on dimensions in dims
                i.e. if dims=[0] on a chess board, elements on different rows would not be able to view each other
        Args:
            X: board embedding (M, D1, ... DN, E)
            src: src embedding (M, S, E)
                we will never mask the source, as this is not a language model
            dims: list of dims to use transformer on, list of indices in [0,N-1]
            collapse_fn: collapses a sequence (M,*,E) to (M,E), usually using netowrks from self
            max_batch: max batch to send through transformer
        """
        # TODO: use this
        dims = list(dims)
        shape = X.shape
        N = len(shape) - 2
        # TODO: maybe do this in a smarter/faster way
        # sends (0,...,N-1) to (0,...*without dims,N-1, dims)
        perm = [i for i in range(N) if i not in dims] + dims
        # inverse permutation to that
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

        cls = self.cls_enc(torch.zeros((X.shape[0], 1),
                                       dtype=torch.long,
                                       device=self.device,
                                       ))
        # shape (*,1,E)

        cls_added = torch.concatenate((cls, X), dim=1)
        # shape (*,1+D{dims_1}*...*D{dims_k},E)

        output = self.transforward(X=cls_added, src=src, max_batch=max_batch)
        # shape (*,1+D{dims_1}*...*D{dims_k},E)

        cls, X = output[:, 0, :], output[:, 1:, :]
        # shape (*,E) and (*,D{dims_1}*...*D{dims_k},E)

        cls = cls.view((shape[0], -1, shape[-1]))
        # shape (M,*,E)
        cls = collapse_fn(cls)
        # shape(M,E)

        X = X.reshape(temp_shape)  # TODO: suspicious, but probably fine to do this
        X = torch.permute(X,
                          [0] + [i + 1 for i in inv_perm] + [N + 1],
                          )
        # returned to original X shape
        return X, cls

    def transforward(self, X, src=None, max_batch=None):
        """
        runs transformer network on an input
        output is same shape as X
        Args:
            X: (M,T,E) sequence
            src: src embedding (M, S, E), or None if only_encoder
            max_batch: maximum batch allowed in the transformer, will split up the calls if X is too large
        """
        if (max_batch is not None) and (X.shape[0] > max_batch):
            output = torch.zeros(X.shape, device=self.device)
            for i in range(0, X.shape[0], max_batch):
                top = min(i + max_batch, X.shape[0])
                if src is None:
                    src_i = src[i:top]
                else:
                    src_i = None
                output[i:top] = self.transforward(X=X[i:top], src=src_i, max_batch=None)  # no need to check max_batch
            return output
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
