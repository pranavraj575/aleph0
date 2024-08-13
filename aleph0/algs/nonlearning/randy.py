import torch

from aleph0.game import SelectionGame
from aleph0.algs.algorithm import Algorithm


class Randy(Algorithm):
    """
    random distribution over possible moves
    """

    def get_policy_value(self, game: SelectionGame, selection_moves=None, special_moves=None):
        if selection_moves is None:
            selection_moves = list(game.get_all_valid_moves())
        if special_moves is None:
            special_moves = list(game.valid_special_moves())
        moves = selection_moves + special_moves
        dist = torch.ones(len(moves))/len(moves)
        return dist, None
