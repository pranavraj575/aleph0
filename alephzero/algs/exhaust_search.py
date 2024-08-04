import torch

from alephzero.game import SubsetGame
from alephzero.algs.algorithm import Algorithm


class Exhasutive(Algorithm):
    """
    minimax search over all possible paths
    outputs a uniform distribution over all 'best moves'

    Note that this does not solve probabilistic games
        can be run with more iterations to get a better approximation of probabilistic games
    """

    def __init__(self, iterations=1):
        """
        Args:
            iterations: number of times to sample to find a distribution
                for deterministic games, this should always be 1, no reason to sample more times
        """
        assert iterations > 0
        self.iterations = iterations

    def choose_distribution(self, all_values, player):
        """
        chooses a distribution based off of expected outcomes of taking each move
        currently a uniform distribution between best outcomes for player
        Args:
            all_values: (N,K) array, where all_values[i] contains the expected payout of each player after taking move i
            player: current player, index to optimize
        Returns: distribution of moves to make based off of this function
        """
        max_player_val = torch.max(all_values[:, player])
        indices = torch.where(torch.eq(all_values[:, player], max_player_val))[0]
        dist = torch.zeros(len(all_values))
        dist[indices] = 1/len(indices)
        return dist

    def minimax_search(self, game: SubsetGame, moves=None):
        """
        minimax search on one game state
        Args:
            game: SubsetGame instance to solve
            moves: moves to check, if None, does all moves
        Returns:
            distribution (N,), values (K,)
        """
        if moves is None:
            moves = list(game.get_all_valid_moves())
        # moves is size N
        all_values = []
        for move in moves:
            next_game: SubsetGame = game.make_move(move)
            if next_game.is_terminal():
                all_values.append(torch.tensor(next_game.get_result(),
                                               dtype=torch.float))
            else:
                _, vals = self.minimax_search(game=next_game,
                                              moves=None,
                                              )
                all_values.append(vals)
        # shape (N,K) for K number of players
        all_values = torch.stack(all_values, dim=0)
        dist = self.choose_distribution(all_values=all_values, player=game.current_player)
        # take the sum of all values, weighted by the distribution
        values = dist.view((1, -1))@all_values
        return dist, values.flatten()

    def get_policy_value(self, game: SubsetGame, moves=None):
        """
        averages runs of self.minimax_search to get optimal policy approximation
            if self.iterations is 1, this is equivalent to self.minimax_search
        """
        dist, values = self.minimax_search(game=game, moves=moves)
        for _ in range(self.iterations - 1):
            dp, vp = self.minimax_search(game=game, moves=moves)
            dist += dp
            values += vp
        dist, values = dist/self.iterations, values/self.iterations
        return dist, values


if __name__ == '__main__':
    from alephzero.examples.tictactoe import Toe

    # if run on initial game, takes a while, then returns that every move is a tying move
    # distribution is uniform over all moves, and value is (.5,.5)
    game = Toe()

    # after taking any move, the result is the same, but runs about 9 times faster
    # distribution is uniform over the four non-losing moves, and value is (.5,.5)
    game = game.make_move(((0, 1),))

    # this is a losing move for player 1.
    # if run on this game, algorithm quickly terminates to return the one winning move for player 0
    # game = game.make_move(((2, 2),))

    print(game.board)
    ex = Exhasutive()
    print(list(game.get_all_valid_moves()))
    print(ex.get_policy_value(game))
