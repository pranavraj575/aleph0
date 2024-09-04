from torch import nn

from aleph0.networks.architect.beginning.input_embedding import InputEmbedding, AutoInputEmbedder
from aleph0.networks.architect.middle.former import Former
from aleph0.networks.architect.middle.transformer import TransFormer, TransFormerEmbedder
from aleph0.networks.architect.middle.cnn import CisFormer
from aleph0.networks.architect.end.policy_value import PolicyValue


class Architect(nn.Module):
    """
    network architecture
    takes input of SelectionGame.observation (board list, positions, vector) and
        possible selection moves and special moves
    outputs a policy distribution and a value estimate
    """

    def __init__(self,
                 input_embedder: InputEmbedding,
                 former: Former,
                 policy_val: PolicyValue,
                 special_moves,
                 device=None,
                 ):
        """
        Args:
            input_embedder: InputEmbedding object to embed input
            former: transformer or cisformer to transform embedded input
            policy_val: goes from former output to policy,value
            special_moves: list of all possible special moves in game
                used to convert from indices to moves and back
        """
        super().__init__()
        self.device = device
        self.input_embedder = input_embedder
        self.former = former
        self.policy_val = policy_val
        self.special_moves = special_moves
        self.special_move_to_idx = {move: i
                                    for i, move in enumerate(self.special_moves)}

    def forward(self,
                observation,
                selection_moves,
                special_moves,
                softmax=False,
                ):
        """
        Args:
            observation: (boards, indexes, vector) observation
            selection_moves: selection mvoes, iterable of list(multidim index)
                usable by policy value net
            special_moves: special moves, iterable of special moves readible by the game being played
            softmax: whether to softmax output
        Returns:
            policy, value, same as policy value net
        """
        embedding, vec_embedding = self.input_embedder.forward(observation=observation)
        embedding, cls_embedding = self.former.forward(embedding, src=vec_embedding)
        return self.policy_val.forward(embedding=embedding,
                                       cls_embedding=cls_embedding,
                                       selection_moves=selection_moves,
                                       special_move_idxs=[self.special_move_to_idx[move]
                                                          for move in special_moves],
                                       softmax=softmax,
                                       )


class AutoArchitect(Architect):
    def __init__(self,
                 sequence_dim,
                 selection_size,
                 former: Former,
                 additional_vector_dim,
                 special_moves,
                 num_players,
                 underlying_set_shapes,
                 underlying_set_sizes=None,
                 encoding_nums=None,
                 base_periods_pre_exp=None,
                 embedding_dim=256,
                 device=None,
                 ):
        """
        Args:
            sequence_dim: dim of sequences expected in input (i.e. an 8x8 board is associated with a 2d seq)
            selection_size: number of indices to select
            additional_vector_dim: dimension of additional vector to add
            special_moves: all possible special moves in game
            num_players: number of players in game
            underlying_set_shapes: shapes of underlying sets of each board
            underlying_set_sizes: sizes of each underylying set, if discreete (i.e. shape ())
                used to create default one hot embeddings
            encoding_nums: encoding nums to use in PositionalAppender
                if unspecified, uses ClassicPositionalEncoding
            base_periods_pre_exp: to be used in PositionalAppender
            embedding_dim: embedding dim to use
            device: device to put stuff on
        """
        if encoding_nums is not None:
            assert len(encoding_nums) == sequence_dim
        if base_periods_pre_exp is not None:
            assert len(base_periods_pre_exp) == sequence_dim
        input_embedder = AutoInputEmbedder(embedding_dim=embedding_dim,
                                           sequence_dim=sequence_dim,
                                           underlying_set_shapes=underlying_set_shapes,
                                           underlying_set_sizes=underlying_set_sizes,
                                           additional_vector_dim=additional_vector_dim,
                                           final_embedding_dim=embedding_dim,
                                           encoding_nums=encoding_nums,
                                           base_periods_pre_exp=base_periods_pre_exp,
                                           )
        embedding_dim = input_embedder.embedding_dim
        policy_val = PolicyValue(embedding_dim=embedding_dim,
                                 selection_size=selection_size,
                                 num_special_moves=len(special_moves),
                                 num_players=num_players,
                                 policy_hidden_layers=[embedding_dim*4],
                                 value_hidden_layers=[embedding_dim*4],
                                 special_hidden_layers=[embedding_dim*4],
                                 )
        super().__init__(input_embedder=input_embedder,
                         former=former,
                         policy_val=policy_val,
                         special_moves=special_moves,
                         device=device,
                         )


class AutoTransArchitect(AutoArchitect):
    def __init__(self,
                 sequence_dim,
                 selection_size,
                 additional_vector_dim,
                 special_moves,
                 num_players,
                 underlying_set_shapes,
                 underlying_set_sizes=None,
                 encoding_nums=None,
                 base_periods_pre_exp=None,
                 embedding_dim=256,
                 dim_feedforward=1024,
                 dropout=.1,
                 nhead=4,
                 num_board_layers=6,
                 num_vector_layers=6,
                 device=None,
                 ):
        """
        Args:
            sequence_dim: dim of sequences expected in input (i.e. an 8x8 board is associated with a 2d seq)
            selection_size: number of indices to select
            additional_vector_dim: dimension of additional vector to add
            special_moves: all possible special moves in game
            num_players: number of players in game
            underlying_set_shapes: shapes of underlying sets of each board
            underlying_set_sizes: sizes of each underylying set, if discreete (i.e. shape ())
                used to create default one hot embeddings
            encoding_nums: encoding nums to use in PositionalAppender
                if unspecified, uses ClassicPositionalEncoding
            base_periods_pre_exp: to be used in PositionalAppender
            embedding_dim: embedding dim to use
            dim_feedforward: feedforward dim in each transformer layer
            nhead: number of heads in transformer
            num_board_layers: number of layers in trans decoder for board embedding
            num_vector_layers: number of layers in trans encoder for vector embedding
            device: device to put stuff on
        """
        if additional_vector_dim == 0:
            former = TransFormerEmbedder(embedding_dim=embedding_dim,
                                         nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         num_layers=num_board_layers,
                                         dropout=dropout,
                                         device=device,
                                         )
        else:
            former = TransFormer(embedding_dim=embedding_dim,
                                 nhead=nhead,
                                 dim_feedforward=dim_feedforward,
                                 num_encoder_layers=num_vector_layers,
                                 num_decoder_layers=num_board_layers,
                                 dropout=dropout,
                                 device=device,
                                 )
        super().__init__(
            sequence_dim=sequence_dim,
            selection_size=selection_size,
            former=former,
            additional_vector_dim=additional_vector_dim,
            special_moves=special_moves,
            num_players=num_players,
            underlying_set_shapes=underlying_set_shapes,
            underlying_set_sizes=underlying_set_sizes,
            encoding_nums=encoding_nums,
            base_periods_pre_exp=base_periods_pre_exp,
            embedding_dim=embedding_dim,
            device=device,
        )


class AutoCisArchitect(AutoArchitect):
    """
    cnn transformation as main body
    probably better for fixed size boards
    """

    def __init__(self,
                 sequence_dim,
                 selection_size,
                 additional_vector_dim,
                 special_moves,
                 num_players,
                 underlying_set_shapes,
                 underlying_set_sizes=None,
                 encoding_nums=None,
                 base_periods_pre_exp=None,
                 embedding_dim=256,
                 num_residuals=6,
                 kernel=None,
                 middle_dim=None,
                 collapse_hidden_layers=None,
                 device=None,
                 ):
        """
        Args:
            sequence_dim: dim of sequences expected in input (i.e. an 8x8 board is associated with a 2d seq)
            selection_size: number of indices to select
            additional_vector_dim: dimension of additional vector to add
            special_moves: all possible special moves in game
            num_players: number of players in game
            underlying_set_shapes: shapes of underlying sets of each board
            underlying_set_sizes: sizes of each underylying set, if discreete (i.e. shape ())
                used to create default one hot embeddings
            encoding_nums: encoding nums to use in PositionalAppender
                if unspecified, uses ClassicPositionalEncoding
            base_periods_pre_exp: to be used in PositionalAppender
            embedding_dim: embedding dim to use
            kernel: multidim kernel to use in CNN
                if none, uses (3,3,..,3)
            middle_dim: middle dimenison to use in CNN resnet
            collapse_hidden_layers: hidden layers to use in collapse layer (if none, uses [4*embedding_dim])
            device: device to put stuff on
        """
        if collapse_hidden_layers is None:
            collapse_hidden_layers = [4*embedding_dim]
        if kernel is None:
            kernel = tuple(3 for _ in range(sequence_dim))
        former = CisFormer(embedding_dim=embedding_dim,
                           num_residuals=num_residuals,
                           kernel=kernel,
                           middle_dim=middle_dim,
                           collapse_hidden_layers=collapse_hidden_layers,
                           )
        super().__init__(
            sequence_dim=sequence_dim,
            selection_size=selection_size,
            former=former,
            additional_vector_dim=additional_vector_dim,
            special_moves=special_moves,
            num_players=num_players,
            underlying_set_shapes=underlying_set_shapes,
            underlying_set_sizes=underlying_set_sizes,
            encoding_nums=encoding_nums,
            base_periods_pre_exp=base_periods_pre_exp,
            embedding_dim=embedding_dim,
            device=device,
        )
