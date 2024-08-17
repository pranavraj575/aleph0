from aleph0.networks.architect.architect import Architect, AutoArchitect

from aleph0.networks.architect.beginning.input_embedding import InputEmbedding, AutoInputEmbedder

from aleph0.networks.architect.beginning.pos_enc import (AbstractPositionalEncoding,
                                                         IdentityPosititonalEncoding,
                                                         ClassicPositionalEncoding,
                                                         PositionalAppender,
                                                         )
from aleph0.networks.architect.beginning.board_embedding import (AutoBoardSetEmbedder,
                                                                 BoardSetEmbedder,
                                                                 BoardEmbedder,
                                                                 LinearEmbedder,
                                                                 FlattenEmbedder,
                                                                 FlattenAndLinearEmbedder,
                                                                 PieceEmbedder,
                                                                 OneHotEmbedder,
                                                                 )

from aleph0.networks.architect.middle.former import Former
from aleph0.networks.architect.middle.transformer import TransFormer
from aleph0.networks.architect.middle.cnn import CisFormer
from aleph0.networks.architect.middle.chainformer import ChainFormer

from aleph0.networks.architect.end.policy_value import PolicyValue

__all__ = [
    'Architect',
    'AutoArchitect',

    "InputEmbedding",
    "AutoInputEmbedder",

    "AbstractPositionalEncoding",
    "IdentityPosititonalEncoding",
    "ClassicPositionalEncoding",
    "PositionalAppender",

    "AutoBoardSetEmbedder",
    "BoardSetEmbedder",
    "BoardEmbedder",
    "LinearEmbedder",
    "FlattenEmbedder",
    "FlattenAndLinearEmbedder",
    "PieceEmbedder",
    "OneHotEmbedder",

    "Former",
    "TransFormer",
    "CisFormer",
    "ChainFormer",

    "PolicyValue",
]
