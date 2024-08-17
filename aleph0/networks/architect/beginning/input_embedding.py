import torch
from torch import nn

from aleph0.networks.architect.beginning.pos_enc import AbstractPositionalEncoding
from aleph0.networks.architect.beginning.board_embedding import BoardSetEmbedder


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
            observation: (boards, indices, vector)
        Returns:
        """
        boards, indices, vector = observation
        board_embedding = self.board_embed(boards)
        board_embedding = self.pos_enc(board_embedding)
        # board embedding is now (M, D1,...,DN, self.pos_enc.embedding_dim)

        if self.additional_vector_dim > 0:
            shape = tuple(board_embedding.shape[:-1]) + (self.additional_vector_dim,)
            vector = vector.broadcast_to(shape)
            board_embedding = torch.cat((board_embedding, vector), dim=-1)
        return self.out(board_embedding)
