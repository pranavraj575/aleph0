import torch
from torch import nn

from aleph0.networks.architect.beginning.pos_enc import (AbstractPositionalEncoding,
                                                         ClassicPositionalEncoding,
                                                         PositionalAppender,
                                                         )
from aleph0.networks.architect.beginning.board_embedding import BoardSetEmbedder, AutoBoardSetEmbedder


class InputEmbedding(nn.Module):
    """
    takes a SelectionGame observation (boards, positions, vector) and embeds it into a single multi-dimensional seq
    """

    def __init__(self,
                 embedding_dim,
                 device,
                 ):
        """
        Args:
            pos_enc: positional encoding to use for the positions part of observations
            board_embed: BoardSetEmbedder to use to embed boards
            additional_vector_dim: dimension of additional vector to add
            final_embedding_dim: if specified, does a linear map to this dimension output
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

    def forward(self, observation):
        """
        Args:
            observation: (boards, positions, vector)
        Returns:
            encoded board, encoded src
        """
        raise NotImplementedError


class InputEmbeddingAppendVector(InputEmbedding):
    """
    takes a SelectionGame observation (boards, positions, vector) and embeds it into a single multi-dimensional seq
        does the expected thing with boards and poistions, then appends the vector onto the end
    """

    def __init__(self,
                 pos_enc: AbstractPositionalEncoding,
                 board_embed: BoardSetEmbedder,
                 additional_vector_dim,
                 final_embedding_dim=None,
                 device=None,
                 ):
        """
        Args:
            pos_enc: positional encoding to use for the positions part of observations
            board_embed: BoardSetEmbedder to use to embed boards
            additional_vector_dim: dimension of additional vector to add
            final_embedding_dim: if specified, does a linear map to this dimension output
        """
        if final_embedding_dim is not None:
            embedding_dim = final_embedding_dim
        else:
            embedding_dim = pos_enc.embedding_dim + additional_vector_dim
        super().__init__(embedding_dim=embedding_dim,
                         device=device,
                         )

        self.pos_enc = pos_enc
        self.board_embed = board_embed
        self.additional_vector_dim = additional_vector_dim
        if final_embedding_dim is not None:
            self.board_out = nn.Linear(in_features=self.pos_enc.embedding_dim,
                                       out_features=final_embedding_dim,
                                       device=self.device,
                                       )
            self.vec_out = nn.Linear(in_features=additional_vector_dim,
                                     out_features=final_embedding_dim,
                                     device=self.device,
                                     )
        else:
            self.board_out = nn.Identity()
            self.vec_out = nn.Identity()

    def forward(self, observation):
        """
        Args:
            observation: (boards, positions, vector)
        Returns:
            encoded boards, None
        """
        boards, positions, vector = observation
        board_embedding = self.board_embed.forward(boards=boards)
        board_embedding = self.pos_enc.forward(X=board_embedding, positions=positions.to(self.device))
        # board embedding is now (M, D1,...,DN, self.pos_enc.embedding_dim)

        if self.additional_vector_dim > 0:
            shape = tuple(board_embedding.shape[:-1]) + (self.additional_vector_dim,)
            vector = vector.to(self.device).broadcast_to(shape)
            board_embedding = torch.cat((board_embedding, vector), dim=-1)
        return self.board_out.forward(board_embedding), None


class InputEmbeddingIgnoreVector(InputEmbedding):
    """
    takes a SelectionGame observation (boards, positions, vector) and embeds it into a single multi-dimensional seq
        does the expected thing with boards and poistions, then ignores the vector
    """

    def __init__(self,
                 pos_enc: AbstractPositionalEncoding,
                 board_embed: BoardSetEmbedder,
                 final_embedding_dim=None,
                 device=None,
                 ):
        """
        Args:
            pos_enc: positional encoding to use for the positions part of observations
            board_embed: BoardSetEmbedder to use to embed boards
            additional_vector_dim: dimension of additional vector to add
            final_embedding_dim: if specified, does a linear map to this dimension output
        """
        if final_embedding_dim is None:
            embedding_dim = pos_enc.embedding_dim
        else:
            embedding_dim = final_embedding_dim
        super().__init__(embedding_dim=embedding_dim,
                         device=device,
                         )
        self.pos_enc = pos_enc
        self.board_embed = board_embed
        if final_embedding_dim is not None:
            self.board_out = nn.Linear(in_features=self.pos_enc.embedding_dim,
                                       out_features=final_embedding_dim,
                                       device=self.device,
                                       )
        else:
            self.board_out = nn.Identity()

    def forward(self, observation):
        """
        Args:
            observation: (boards, positions, vector)
        Returns:
        """
        boards, positions, vector = observation
        board_embedding = self.board_embed.forward(boards=boards)
        board_embedding = self.pos_enc.forward(X=board_embedding, positions=positions.to(self.device))
        # board embedding is now (M, D1,...,DN, self.pos_enc.embedding_dim)

        return self.board_out.forward(board_embedding), None


class InputEmbeddingMapVector(InputEmbedding):
    """
    takes a SelectionGame observation (boards, positions, vector) and embeds it into a single multi-dimensional seq
        does the expected thing with boards and poistions, then ignores the vector
    """

    def __init__(self,
                 pos_enc: AbstractPositionalEncoding,
                 board_embed: BoardSetEmbedder,
                 vector_map,
                 vector_final_dim=None,
                 final_embedding_dim=None,
                 device=None,
                 ):
        """
        Args:
            pos_enc: positional encoding to use for the positions part of observations
            board_embed: BoardSetEmbedder to use to embed boards
            vector_map: map to apply to additional vector
            vector_final_dim: used to apply a linear layer to, if final_embedding_dim is specified
            final_embedding_dim: if specified, does a linear map to this dimension output
        """
        if final_embedding_dim is None:
            embedding_dim = pos_enc.embedding_dim
        else:
            embedding_dim = final_embedding_dim
        super().__init__(embedding_dim=embedding_dim,
                         device=device,
                         )
        self.pos_enc = pos_enc
        self.board_embed = board_embed
        self.vector_map = vector_map
        if final_embedding_dim is not None:
            self.board_out = nn.Linear(in_features=self.pos_enc.embedding_dim,
                                       out_features=final_embedding_dim,
                                       device=self.device,
                                       )
            self.vec_out = nn.Linear(in_features=vector_final_dim,
                                     out_features=final_embedding_dim,
                                     device=self.device,
                                     )
        else:
            self.board_out = nn.Identity()
            self.vec_out = nn.Identity()

    def forward(self, observation):
        """
        Args:
            observation: (boards, positions, vector)
        Returns:
        """
        boards, positions, vector = observation
        board_embedding = self.board_embed.forward(boards=boards)
        board_embedding = self.pos_enc.forward(X=board_embedding, positions=positions.to(self.device))
        # board embedding is now (M, D1,...,DN, self.pos_enc.embedding_dim)
        vector_embedding = self.vector_map(vector.to(self.device))
        return self.board_out.forward(board_embedding), self.vec_out.forward(vector_embedding)


class VecToSeqEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vec):
        """
        goes from an (M,N) dim vector to
            (M,1,N) dimensitonal
        """
        return vec.unsqueeze(1)


def AutoInputEmbedder(
        embedding_dim,
        sequence_dim,
        additional_vector_dim,
        underlying_set_shapes,
        underlying_set_sizes=None,
        final_embedding_dim=None,
        encoding_nums=None,
        base_periods_pre_exp=None,
        device=None,
):
    """
    Args:
        embedding_dim: embedding dim to use
        sequence_dim: dim of sequences expected in input (i.e. an 8x8 board is associated with a 2d seq)
        additional_vector_dim: dimension of additional vector to add
        underlying_set_shapes: shapes of underlying sets of each board
        underlying_set_sizes: sizes of each underylying set, if discreete (i.e. shape ())
            used to create default one hot embeddings
        encoding_nums: encoding nums to use in PositionalAppender
            if None, uses ClassicPositionalEncoding
        base_periods_pre_exp: to be used in PositionalAppender
    """
    if encoding_nums is not None:
        assert len(encoding_nums) == sequence_dim
    if base_periods_pre_exp is not None:
        assert len(base_periods_pre_exp) == sequence_dim
    board_embed = AutoBoardSetEmbedder(underlying_set_shapes=underlying_set_shapes,
                                       underlying_set_sizes=underlying_set_sizes,
                                       final_embedding_dim=embedding_dim,
                                       device=device,
                                       )
    if encoding_nums is None:
        pos_enc = ClassicPositionalEncoding(embedding_dim=embedding_dim,
                                            sequence_dim=sequence_dim,
                                            device=device,
                                            )
    else:
        pos_enc = PositionalAppender(input_embedding_dim=embedding_dim,
                                     encoding_nums=encoding_nums,
                                     base_periods_pre_exp=base_periods_pre_exp,
                                     device=device,
                                     )
    if additional_vector_dim == 0:
        return InputEmbeddingIgnoreVector(pos_enc=pos_enc,
                                          board_embed=board_embed,
                                          final_embedding_dim=final_embedding_dim,
                                          device=device,
                                          )
    else:
        return InputEmbeddingMapVector(
            pos_enc=pos_enc,
            board_embed=board_embed,
            vector_map=VecToSeqEmbedding(),
            final_embedding_dim=final_embedding_dim,
            vector_final_dim=final_embedding_dim,
            device=device,
        )
