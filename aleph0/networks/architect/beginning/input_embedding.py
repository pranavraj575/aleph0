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
        does the expected thing with boards and poistions, then appends the vector onto the end
    """

    def __init__(self,
                 pos_enc: AbstractPositionalEncoding,
                 board_embed: BoardSetEmbedder,
                 additional_vector_dim,
                 final_embedding_dim=None,
                 ):
        """
        Args:
            pos_enc: positional encoding to use for the positions part of observations
            board_embed: BoardSetEmbedder to use to embed boards
            additional_vector_dim: dimension of additional vector to add
            final_embedding_dim: if specified, does a linear map to this dimension output
        """
        super().__init__()
        self.pos_enc = pos_enc
        self.board_embed = board_embed
        self.additional_vector_dim = additional_vector_dim
        if final_embedding_dim is not None:
            self.out = nn.Linear(in_features=self.pos_enc.embedding_dim + self.additional_vector_dim,
                                 out_features=final_embedding_dim,
                                 )
            self.embedding_dim = final_embedding_dim
        else:
            self.out = nn.Identity()
            self.embedding_dim = self.pos_enc.embedding_dim + self.additional_vector_dim

    def forward(self, observation):
        """
        Args:
            observation: (boards, positions, vector)
        Returns:
        """
        boards, positions, vector = observation
        board_embedding = self.board_embed.forward(boards=boards)
        board_embedding = self.pos_enc.forward(X=board_embedding, positions=positions)
        # board embedding is now (M, D1,...,DN, self.pos_enc.embedding_dim)

        if self.additional_vector_dim > 0:
            shape = tuple(board_embedding.shape[:-1]) + (self.additional_vector_dim,)
            vector = vector.broadcast_to(shape)
            board_embedding = torch.cat((board_embedding, vector), dim=-1)
        return self.out.forward(board_embedding)


class AutoInputEmbedder(InputEmbedding):
    def __init__(self,
                 embedding_dim,
                 sequence_dim,
                 additional_vector_dim,
                 underlying_set_shapes,
                 underlying_set_sizes=None,
                 final_embedding_dim=None,
                 encoding_nums=None,
                 base_periods_pre_exp=None,
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
        board_embed = AutoBoardSetEmbedder(underlying_set_shapes=underlying_set_shapes,
                                           underlying_set_sizes=underlying_set_sizes,
                                           final_embedding_dim=embedding_dim,
                                           )
        if encoding_nums is None:
            pos_enc = ClassicPositionalEncoding(embedding_dim=embedding_dim,
                                                sequence_dim=sequence_dim,
                                                )
        else:
            pos_enc = PositionalAppender(input_embedding_dim=embedding_dim,
                                         encoding_nums=encoding_nums,
                                         base_periods_pre_exp=base_periods_pre_exp,
                                         )
        super().__init__(pos_enc=pos_enc,
                         board_embed=board_embed,
                         additional_vector_dim=additional_vector_dim,
                         final_embedding_dim=final_embedding_dim)
