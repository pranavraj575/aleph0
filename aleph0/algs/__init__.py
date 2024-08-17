from aleph0.algs.nonlearning import Human, Exhasutive, Randy, MCTS
from aleph0.algs.learning import DQNAlg, DQNAlg_from_game
from aleph0.algs.play_game import play_game
from aleph0.algs.algorithm import Algorithm

__all__ = [
    'Algorithm',
    'DQNAlg',
    "DQNAlg_from_game",

    'Exhasutive',
    'Human',
    'Randy',
    "MCTS",

    'play_game'
]
