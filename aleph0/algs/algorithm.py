import os, pickle

from aleph0.game import SelectionGame


class Algorithm:
    def __init__(self):
        self.info = dict()

    def get_policy_value(self, game: SelectionGame, moves=None):
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
                or None if not calculated
        """
        raise NotImplementedError

    def save(self, save_dir):
        """
        save to specified dir
        Args:
            save_dir: save dir
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        f = open(os.path.join(save_dir, 'info.pkl'), 'wb')
        pickle.dump(self.info, f)
        f.close()

    def load(self, save_dir):
        """
        loads from specified dir
        Args:
            save_dir: save dir
        """

        f = open(os.path.join(save_dir, 'info.pkl'), 'rb')
        self.info.update(pickle.load(f))
        f.close()
