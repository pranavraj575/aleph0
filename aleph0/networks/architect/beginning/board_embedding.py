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


class DiscretePieceEnc(BoardEmbedder):
    """
    embeds boards of discrete pieces
    """


class PieceEmbedder(DiscretePieceEnc):
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


class OneHotEmbedder(DiscretePieceEnc):
    """
    embeds boards of discrete pieces as one-hot vectors
    """

    def __init__(self, piece_count):
        super().__init__(embedding_dim=piece_count)

    def forward(self, board):
        """
        Args:
            board: (M, D1, ..., DN, *)
        Returns: (M, D1, ..., DN, E)
        """
        return torch.nn.functional.one_hot(board, num_classes=self.embedding_dim).float()


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


class AutoBoardSetEmbedder(BoardSetEmbedder):
    """
    automatically makes a board set embedder given the observation piece shapes
    """

    def __init__(self,
                 underlying_set_shapes,
                 underlying_set_sizes=None,
                 default_discrete_piece_classes=None,
                 default_discrete_piece_args=None,
                 default_vector_piece_classes=None,
                 default_vector_piece_args=None,
                 board_embedding_list=None,
                 final_embedding_dim=None
                 ):
        """
        Args:
            underlying_set_shapes: shapes of underlying sets of each board
            underlying_set_sizes: sizes of each underylying set, if discreete (i.e. shape ())
                used to create default one hot embeddings
            board_embedding_list: board embedding list, if any defined
                i.e. array of Nones will create default board embeddings
            default_discrete_piece_classes: list of classes to use to embed discrete pieces
                (if None, uses 1-hot vectors)
            default_discrete_piece_args: list of dict of args to use for each discrete piece class
            default_vector_piece_classes: list of classes to use to embed vector pieces
                (if None, flattens vectors and uses as is)
            default_vector_piece_args: list of dict of args to use for each vector piece class
        """
        req_len = len(underlying_set_shapes)
        if board_embedding_list is None:
            board_embedding_list = [None for _ in range(req_len)]

        if len(board_embedding_list) < req_len:
            board_embedding_list.extend(
                [None for _ in range(req_len - len(board_embedding_list))]
            )

        if underlying_set_sizes is None:
            underlying_set_sizes = [None for _ in range(req_len)]
        if len(underlying_set_sizes) < req_len:
            underlying_set_sizes.extend(
                [None for _ in range(req_len - len(underlying_set_sizes))]
            )

        if default_discrete_piece_classes is None:
            default_discrete_piece_classes = [None for _ in range(req_len)]
        if len(default_discrete_piece_classes) < req_len:
            default_discrete_piece_classes.extend(
                [None for _ in range(req_len - len(default_discrete_piece_classes))]
            )

        if default_discrete_piece_args is None:
            default_discrete_piece_args = [None for _ in range(req_len)]
        if len(default_discrete_piece_args) < req_len:
            default_discrete_piece_args.extend(
                [None for _ in range(req_len - len(default_discrete_piece_args))]
            )

        if default_vector_piece_classes is None:
            default_vector_piece_classes = [None for _ in range(req_len)]
        if len(default_vector_piece_classes) < req_len:
            default_vector_piece_classes.extend(
                [None for _ in range(req_len - len(default_vector_piece_classes))]
            )

        if default_vector_piece_args is None:
            default_vector_piece_args = [None for _ in range(req_len)]
        if len(default_vector_piece_args) < req_len:
            default_vector_piece_args.extend(
                [None for _ in range(req_len - len(default_vector_piece_args))]
            )

        for i, (underlying_set_shape,
                underlying_set_size,
                DiscretePieceClass,
                discrete_piece_kwargs,
                VectorPieceClass,
                vector_kwargs,
                ) in enumerate(
            zip(underlying_set_shapes,
                underlying_set_sizes,
                default_discrete_piece_classes,
                default_discrete_piece_args,
                default_vector_piece_classes,
                default_vector_piece_args,
                )):
            if board_embedding_list[i] is None:
                if underlying_set_shape == ():
                    if DiscretePieceClass is None:
                        assert isinstance(underlying_set_size,
                                          int)  # need to specify this if using default discrete embeddings

                        DiscretePieceClass = OneHotEmbedder
                        discrete_piece_kwargs = {'piece_count': underlying_set_size}
                    if discrete_piece_kwargs is None:
                        discrete_piece_kwargs = dict()
                    board_embedding_list[i] = DiscretePieceClass(**discrete_piece_kwargs)
                else:
                    if VectorPieceClass is None:
                        VectorPieceClass = FlattenEmbedder
                        vector_kwargs = {'input_shape': underlying_set_shape}
                    if vector_kwargs is None:
                        vector_kwargs = dict()
                    board_embedding_list[i] = VectorPieceClass(**vector_kwargs)
        super().__init__(board_embedding_list=board_embedding_list,
                         final_embedding_dim=final_embedding_dim,
                         )


if __name__ == '__main__':
    from aleph0.examples.chess import Chess5d
    from aleph0.examples.chess.game import P

    game = Chess5d()
    for _ in range(10):
        game = game.make_move(next(game.get_all_valid_moves()))
    print(game.observation_shape)
    embedder0 = AutoBoardSetEmbedder(underlying_set_shapes=game.get_underlying_set_shapes(),
                                     underlying_set_sizes=game.underlying_set_sizes(),
                                     )
    boards, _, _ = game.batch_obs
    # embedding_dim should be sum of all possible pieces
    # (total pieces + blocked piece) + 3 boolean variables
    print('predicted embedding dim', (P.TOTAL_PIECES + 1) + 2 + 2 + 2)
    print(embedder0.forward(boards).shape)
    print()

    embedder2 = AutoBoardSetEmbedder(
        underlying_set_shapes=game.get_underlying_set_shapes(),
        default_discrete_piece_classes=[PieceEmbedder for _ in range(len(game.get_underlying_set_shapes()))],
        default_discrete_piece_args=[{'embedding_dim': 69,
                                      'piece_count': P.TOTAL_PIECES + 1}
                                     ] + [{'embedding_dim': 69,
                                           'piece_count': 2}]*3
    )

    print('predicted embedding dim', 69*4)
    print(embedder2.forward(boards).shape)
    print()

    embedder3 = AutoBoardSetEmbedder(underlying_set_shapes=game.get_underlying_set_shapes(),
                                     underlying_set_sizes=game.underlying_set_sizes(),
                                     final_embedding_dim=69,
                                     )
    boards, _, _ = game.batch_obs
    # embedding_dim should be sum of all possible pieces
    # (total pieces + blocked piece) + 3 boolean variables
    print('predicted embedding dim', 69)
    print(embedder3.forward(boards).shape)
