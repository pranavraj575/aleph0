"""
convolutional networks
treats input board (batch size, D1, ..., input_dim) as a 4d array, and convolves appropriately
"""
import torchConvNd
import torch
from torch import nn
from aleph0.networks.collapse import Collapse


class CisToTransPerm(nn.Module):
    """
    permutes from convolution order (batch size, channels, D1, D2, ...)
    to transformer order (batch size, D1, D2, ..., channels)
    """

    def __init__(self, num_dims):
        super().__init__()
        num_dims = num_dims + 2
        # assume X has k+1 dimensions (0, ...,k)
        perm = list(range(num_dims))
        perm[-1] = 1
        perm[1:-1] = range(2, num_dims)
        # perm should be [0,2,3,...,k,1]
        self.perm = perm

    def forward(self, X):
        return X.permute(self.perm)


class TransToCisPerm(nn.Module):
    """
    permutes from transformer order (batch size, D1, D2, ..., channels)
    to convolution order (batch size, channels, D1, D2, ...)
    """

    def __init__(self, num_dims):
        """
        Args:
            num_dims: D1,...,DN, does not include batch size or embedding dim
        """
        super().__init__()
        num_dims = num_dims + 2

        k = num_dims - 1
        # assume X has k+1 dimensions (0, ..., k)
        # includes batch dimension
        # this list is (0, k, 1, ..., k-1)
        perm = list(range(num_dims))
        perm[1] = k
        perm[2:] = range(1, k)
        self.perm = perm

    def forward(self, X):
        return X.permute(self.perm)


class ResBlock(nn.Module):
    """
    adds residuals to the embedding with CNN
    uses two convolutions and adds the result to the input
    """

    def __init__(self, num_channels: int, kernel, middle_channels=None):
        """
        if middle_channels is None, use num_channels in the middle
        kernel must be all odd numbers so that we can keep the dimensions the same
        """
        super(ResBlock, self).__init__()
        for k in kernel:
            if not k%2:
                raise Exception('kernel must be only odd numbers')

        if middle_channels is None:
            middle_channels = num_channels
        self.num_channels = num_channels
        self.middle_channels = middle_channels
        stride = [1 for _ in kernel]
        padding = [(k - 1)//2 for k in kernel]

        self.conv1 = torchConvNd.ConvNd(num_channels,
                                        middle_channels,
                                        list(kernel),
                                        stride=stride,
                                        padding=padding,
                                        )
        self.conv1_param = nn.ParameterList(self.conv1.parameters())
        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = torchConvNd.ConvNd(middle_channels,
                                        num_channels,
                                        list(kernel),
                                        stride=stride,
                                        padding=padding,
                                        )
        self.conv2_param = nn.ParameterList(self.conv2.parameters())
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.relu2 = nn.ReLU()

    def forward(self, X):
        """
        :param X: shaped (batch size, input channels, D1, D2, ...)
        :return: (batch size, output channels, D1, D2, ...)
        """
        # (batch size, num channels, D1, D2, ...)
        _X = X

        batch_size = X.shape[0]
        # other_dimensions is (D1, D2, ...)

        # (batch size, middle channels, D1, D2, ...)
        X = self.conv1(X)
        inter_shape = X.shape
        # (batch size, middle channels, M) where M is the product of the dimensions
        X = X.view(batch_size, self.middle_channels, -1)
        X = self.bn1(X)
        # (batch size, middle channels, D1, D2, ...)
        X = X.view(inter_shape)
        X = self.relu1(X)

        # (batch size, num channels, D1, D2, ...)
        X = self.conv2(X)
        inter_shape = X.shape
        # (batch size, num channels, M)
        X = X.view(batch_size, self.num_channels, -1)
        X = self.bn2(X)
        # (batch size, num channels, D1, D2, ...)
        X = X.view(inter_shape)
        return self.relu2(_X + X)


class CisArchitect(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_residuals,
                 kernel,
                 middle_dim=None,
                 collapse_hidden_layers=None,
                 ):
        """
        pastes a bunch of CNNs together

        """
        super().__init__()

        self.perm1 = TransToCisPerm(num_dims=len(kernel))

        self.layers = nn.ModuleList([
            ResBlock(num_channels=embedding_dim, kernel=kernel, middle_channels=middle_dim) for _ in
            range(num_residuals)
        ])
        # this permutation is nessary for collapsing, as collapse keeps the last dimension
        self.perm2 = CisToTransPerm(num_dims=len(kernel))

        self.collapse = Collapse(embedding_dim=embedding_dim,
                                 hidden_layers=collapse_hidden_layers,
                                 )

    def forward(self, X):
        """
        note: batch size is kept for legacy, it will probably be 1
        :param X: (batch size, D1, ..., Dk, embedding_dim)
        :return: (batch size, M), (batch size, 1): a policy (probability distribution) and a value
        """
        # X is (batch size, D1, D2, ..., embedding_dim)

        # now (batch size, embedding_dim, D1, D2, ...)
        X = self.perm1(X)

        for layer in self.layers:
            X = layer(X)

        # (batch size, D1, D2, ..., embedding dim)
        X = self.perm2(X)
        return X, self.collapse(X)


if __name__ == '__main__':
    # try teaching model to distinguaish its collapsed value from transformed random noise input
    # this is considerably more difficult than the trans architect, as the collapsed value is simply a weighted sum
    # of each other value

    # roughly what the model learns seems to be to designate one trash input to set to 1
    # this will be weighted as 1 and become the cls output
    # the rest of the values will be the correct value for the out output

    embedding_dim = 16
    out_dim = 1
    cis = CisArchitect(embedding_dim=embedding_dim,
                       num_residuals=2,
                       kernel=(3, 3, 3, 3),
                       )
    test_out = None
    cls_out = None
    optim = torch.optim.Adam(cis.parameters())
    losses = []
    out_values = []
    cls_values = []
    for i in range(4000):
        overall_loss = torch.zeros(1)
        overall_out = torch.zeros(1)
        overall_cls = torch.zeros(1)
        batch_size = 1
        for _ in range(batch_size):
            test = torch.rand((1, 1, 1, 1, 4, embedding_dim))
            test_out, cls_out = cis(test)
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
    print(test_out)
    print(cls_out)
