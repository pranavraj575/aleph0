import torch, math
from torch import nn


class AbstractPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X, positions):
        """
        can also drop batch from all args
        Args:
            X: input, shape (batch_size, D1, ..., Dn, initial)
            positions: indexes of each dimension, shape (batch_size, D1, ..., DN, N)
        Returns:
            shape (batch_size, D1, ..., Dn, self.embedding_dim) output
        """
        raise NotImplementedError


class IdentityPosititonalEncoding(AbstractPositionalEncoding):
    """
    does nothing
    """

    def __init__(self, embedding_dim):
        super().__init__(embedding_dim=embedding_dim)


class ClassicPositionalEncoding(AbstractPositionalEncoding):
    def __init__(self, embedding_dim, sequence_dim, base_periods_pre_exp=None):
        """

        Args:
            embedding_dim: dim of state embedding
            sequence_dim: dim of sequences expected in input (i.e. an 8x8 board is associated with a 2d seq)
            base_periods_pre_exp: base period to use for each dimension
                periods are calculated by e^{torch.arange(embedding_dim//2)*base_period_pre_exp}
                i.e. if seq is 1-d, using a base period of -log(2) means we embed positions with periods of
                        1,1/2,1/4,...
                by default uses -2*log(10000)/embedding_dim
        """
        super().__init__(embedding_dim=embedding_dim)
        self.N = sequence_dim
        if base_periods_pre_exp is None:
            base_periods_pre_exp = [-2*math.log(10000.)/embedding_dim for _ in range(self.N)]
        self.base_periods_pre_exp = base_periods_pre_exp

        # (N, embedding_dim//2) array of periods to use on each dimension
        periods = torch.stack([torch.exp(torch.arange(embedding_dim//2)*bppe)
                               for bppe in self.base_periods_pre_exp],
                              dim=0)
        self.register_buffer('periods', periods)
        # if embedding dim is odd, we must mess with stuff later
        self.odd_embedding = (embedding_dim%2)

    def forward(self, X, positions):
        """
        X has shape (batch_size, D1, ..., Dn, embedding_dim)
        can also drop batch from all args
        Args:
            X: input, shape (batch_size, D1, ..., Dn, embedding_dim)
            positions: indexes of each dimension, shape (batch_size, D1, ..., DN, N)
        """
        # (batch_size, D1, ..., DN, N, embedding_dim//2)
        pre_periodic = positions.unsqueeze(-1)*self.periods
        # full_encoding (batch_size, D1, ..., DN, N, embedding_dim)
        if self.odd_embedding:
            # zero_shape is (batch_size, D1, ..., DN, N, 1), it adds one to the encoding dim to fix oddness
            zero_shape = list(pre_periodic.shape)
            zero_shape[-1] = 1
            full_encoding = torch.cat([torch.sin(pre_periodic),
                                       torch.cos(pre_periodic),
                                       torch.zeros(zero_shape),
                                       ],
                                      dim=-1)
        else:
            full_encoding = torch.cat([torch.sin(pre_periodic),
                                       torch.cos(pre_periodic),
                                       ], dim=-1)
        # average out all the encodings, size (batch_size, D1, ..., DN, embedding_dim)
        pos_enc = torch.mean(full_encoding, dim=-2)

        return X + pos_enc


class PositionalAppendingLayer(AbstractPositionalEncoding):
    def __init__(self, input_embedding_dim, encoding_nums, base_periods_pre_exp=None):
        """

        Args:
            encoding_nums: tuple of ints
                number of different frequencies to encode each dimension with
            base_periods_pre_exp: base period to use for each dimension
                periods are calculated by e^{torch.arange(encoding_nums)*base_period_pre_exp}
                i.e. if seq is 1-d, using a base period of -log(2) means we embed positions with periods of
                    1,1/2,1/4,...
                by default uses -log(10000)/encoding_num
        """
        super().__init__(embedding_dim = input_embedding_dim + 2*sum(self.encoding_nums))
        self.encoding_nums = encoding_nums
        if base_periods_pre_exp is None:
            base_periods_pre_exp = [-math.log(10000.)/encoding_num for encoding_num in encoding_nums]
        self.base_periods_pre_exp = base_periods_pre_exp

    def forward(self, X, positions):
        """
        X has shape (batch_size, D1, ..., Dn, initial dim)

        The output will have shape (batch_size, D1, ..., DN, initial_embedding + additional_output)

        can also drop batch from all args
        Args:
            X: input, shape (batch_size, D1, ..., Dn, initial_embedding)
            positions: indexes of each dimension, shape (batch_size, D1, ..., DN, N)
        """
        items = [X]
        posshep = positions.shape

        # (N, batch_size, D1, ..., DN, 1)
        # or (N, D1, ..., DN, 1)
        perm_pos = positions.permute([(i - 1)%len(posshep) for i in range(len(posshep))]).unsqueeze(-1)
        for dimension, (encoding_num, bppe) in enumerate(zip(self.encoding_nums, self.base_periods_pre_exp)):
            # of size (encoding_num,)
            periods = torch.exp(torch.arange(encoding_num)*bppe)
            # (N, batch_size, D1, ..., DN, encoding_num)
            # or (N, D1, ..., DN, encoding_num)
            pre_periodic = perm_pos[dimension]*periods

            items.append(torch.sin(pre_periodic))
            items.append(torch.cos(pre_periodic))

        # (N, batch_size, D1, ..., DN, self.additional_output)
        # or (N, D1, ..., DN, self.additional_output)
        return torch.cat(items, dim=-1)


if __name__ == '__main__':
    input_embedding_dim = 7
    appending_layer = PositionalAppendingLayer(input_embedding_dim=input_embedding_dim,
                                               encoding_nums=(3, 3, 3),
                                               base_periods_pre_exp=(-math.log(2), -math.log(2), -math.log(2))
                                               )
    xyz = .69*torch.ones(1, 1, 3, input_embedding_dim)
    x = torch.arange(xyz.shape[0]).reshape(-1, 1, 1)
    x = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
    y = torch.arange(xyz.shape[1]).reshape(1, -1, 1)
    y = torch.stack([torch.zeros_like(y), y, torch.zeros_like(y)], dim=-1)
    z = torch.arange(xyz.shape[2]).reshape(1, 1, -1)
    z = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)
    print(appending_layer(xyz, x + y + z))
    print(torch.arccos(appending_layer(xyz, x + y + z)))

    encoding_layer = ClassicPositionalEncoding(embedding_dim=input_embedding_dim,
                                               sequence_dim=3,
                                               base_periods_pre_exp=(-math.log(2), -math.log(2), -math.log(2))
                                               )
    print(encoding_layer(xyz, x + y + z))
    print(torch.arcsin(3*encoding_layer(xyz, x + y + z)))
