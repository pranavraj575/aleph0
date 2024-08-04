import torch

from alephzero.game import SubsetGame


class Algorithm:
    def get_policy_value(self, game: SubsetGame, moves=None):
        """
        gets the distribution of best moves from the state of game, as well as the value for each player
        requires that game is not at a terminal state
        Args:
            game: SubsetGame instance with K players
            moves: list of valid moves to inspect (size N)
                if None, uses game.get_all_valid_moves()
        Returns:
            array of size N that determines the calculated probability of taking each move,
                in order of moves given, or game.get_all_valid_moves()
            array of size K in game that determines each players expected payout
        """
        raise NotImplementedError
