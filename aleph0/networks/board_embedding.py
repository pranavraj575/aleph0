import torch
from torch import nn


class BoardEmbedder(nn.Module):
    """
    embeds a board (M, D1, ..., DN, *) into embedding dim (M, D1, ..., DN, E)
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, board):
        raise NotImplementedError


class PieceEmbedder(BoardEmbedder):
    """
    embeds boards of discrete pieces
    """

    def __init__(self, embedding_dim, piece_count):
        super().__init__(embedding_dim=embedding_dim)
        self.embedder = nn.Embedding(num_embeddings=piece_count, embedding_dim=self.embedding_dim)

    def forward(self, board):
        """
        Args:
            board: (M, D1, ..., DN, *)
        Returns: (M, D1, ..., DN, E)
        """
        return self.embedder(board)


class LinearEmbedder(BoardEmbedder):
    """
    embeds boards of vectors through a simple linear map
    """

    def __init__(self, embedding_dim, input_dim):
        super().__init__(embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=input_dim, out_features=embedding_dim)

    def forward(self, board):
        """
        Args:
            board: (M, D1, ..., DN, input_dim)
        Returns: (M, D1, ..., DN, E)
        """
        return self.linear(board)


class FlattenEmbedder(BoardEmbedder):
    """
    embeds boards of vectors by flattening them
    """

    def __init__(self, input_shape, embedding_dim=None):
        """

        Args:
            input_shape: shape of each piece
                i.e. board is shape (M, D1, ..., DN, input_shape)
            embedding_dim: product of input shape
                if None, calculates this
        """
        if embedding_dim is None:
            embedding_dim = 1
            for item in input_shape:
                embedding_dim = int(item*embedding_dim)
        super().__init__(embedding_dim=embedding_dim)
        self.flatten = nn.Flatten(start_dim=-len(input_shape), end_dim=-1)

    def forward(self, board):
        """
        Args:
            board: (M, D1, ..., DN, input_dim)
        Returns: (M, D1, ..., DN, E)
        """
        return self.flatten(board)


class FlattenAndLinearEmbedder(BoardEmbedder):
    """
    first flattens a piece, then linearly maps it to embedding_dim
    """

    def __init__(self, input_shape, embedding_dim):
        super().__init__(embedding_dim)
        self.flatten = FlattenEmbedder(input_shape=input_shape)
        flattened_size = self.flatten.embedding_dim
        self.linear = nn.Linear(in_features=flattened_size, out_features=embedding_dim)

    def forward(self, board):
        board = self.flatten(board)
        return self.linear(board)


class BoardSetEmbedder(nn.Module):
    """
    embeds a list of (M, D1, ..., DN, *) boards
    The * may be different for each board

    uses board_embedding_list to embed each board, combines them,
        maybe does a final linear map to the desired input dim
    """

    def __init__(self, board_embedding_list, final_embedding_dim=None):
        """
        Args:
            board_embedding_list: list of BoardEmbedder objects, same size as number of boards
            final_embedding_dim: final embedding dim, if specified, does a final linear embedding from the concatenated board embeddings
        """
        super().__init__()
        self.board_embedders = nn.ModuleList(board_embedding_list)
        self.cat_input = sum(board_embedder.embedding_dim for board_embedder in board_embedding_list)
        if final_embedding_dim:
            self.embedding_dim = final_embedding_dim
            self.final_embedding = nn.Linear(in_features=self.cat_input, out_features=final_embedding_dim)
        else:
            self.embedding_dim = self.cat_input
            self.final_embedding = nn.Identity()

    def forward(self, boards):
        """
        Args:
            boards: list of (M, D1, ..., DN, *) board
        Returns:
            (M, D1, ..., DN, E)
        """
        board_embeddings = [be.forward(board)
                            for be, board in zip(self.board_embedders, boards)]
        board_cat = torch.cat(board_embeddings, dim=-1)
        return self.final_embedding(board_cat)
