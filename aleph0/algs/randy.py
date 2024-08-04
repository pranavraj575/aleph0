import torch

from aleph0.game import SubsetGame
from aleph0.algs.algorithm import Algorithm


class Randy(Algorithm):
    """
    random distribution over possible moves
    """

    def get_policy_value(self, game: SubsetGame, moves=None):
        if moves is None:
            moves = list(game.get_all_valid_moves())
        dist = torch.ones(len(moves))/len(moves)
        return dist, None

