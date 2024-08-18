from aleph0.algs.algorithm import Algorithm
from aleph0.algs.learning import AlephZero, DQNAlg, DQNAlg_from_game, Human
from aleph0.algs.nonlearning import Exhasutive, Randy, MCTS
from aleph0.algs.play_game import play_game

__all__ = [
    'Algorithm',

    'AlephZero',
    'DQNAlg',
    "DQNAlg_from_game",
    'Human',

    'Exhasutive',
    'Randy',
    "MCTS",

    'play_game'
]
