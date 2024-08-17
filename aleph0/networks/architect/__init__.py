from aleph0.networks.architect.architect import Architect

from aleph0.networks.architect.beginning.input_embedding import InputEmbedding
from aleph0.networks.architect.beginning.pos_enc import (AbstractPositionalEncoding,
                                                         IdentityPosititonalEncoding,
                                                         ClassicPositionalEncoding,
                                                         PositionalAppendingLayer,
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

    "InputEmbedding",

    "AbstractPositionalEncoding",
    "IdentityPosititonalEncoding",
    "ClassicPositionalEncoding",
    "PositionalAppendingLayer",

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