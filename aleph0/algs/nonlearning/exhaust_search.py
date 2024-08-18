import torch

from aleph0.game import SelectionGame
from aleph0.algs.algorithm import Algorithm


class Exhasutive(Algorithm):
    """
    minimax search over all possible paths
    outputs a uniform distribution over all 'best moves'

    Note that this does not solve probabilistic games
        can be run with more iterations to get a better approximation of probabilistic games
    """

    def __init__(self, iterations=1, cache_depth_rng=(-1, -1)):
        """
        Args:
            iterations: number of times to sample to find a distribution
                for deterministic games, this should always be 1, no reason to sample more times
            cache_depth_rng: depth to keep a cache to
                if (0,float('inf')), keeps a cache of all seen observations
                if (0,2), keeps only the 0th and 1st depth boards
        """
        super().__init__()
        assert iterations > 0
        self.iterations = iterations
        self.cache = dict()
        self.cache_depth_rng = cache_depth_rng

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

    def minimax_search(self, game: SelectionGame, moves=None, depth=0):
        if depth < self.cache_depth_rng[0] or depth >= self.cache_depth_rng[1]:
            return self._minimax_search(game=game, moves=moves)
        boards, positions, vec = game.observation
        boards = tuple(tuple(board.flatten().tolist()) for board in boards)
        positions = tuple(positions.flatten().tolist())
        vec = tuple(vec.flatten().tolist())
        obs = boards, positions, vec, game.current_player  # current player is important probably

        if obs not in self.cache:
            self.cache[obs] = self._minimax_search(game=game, moves=moves, depth=depth + 1)
        return self.cache[obs]

    def _minimax_search(self, game: SelectionGame, moves=None, depth=0):
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
            next_game: SelectionGame = game.make_move(move)
            if next_game.is_terminal():
                all_values.append(torch.tensor(next_game.get_result(),
                                               dtype=torch.float))
            else:
                _, vals = self.minimax_search(game=next_game,
                                              moves=None,
                                              depth=depth,
                                              )
                all_values.append(vals)
        # shape (N,K) for K number of players
        all_values = torch.stack(all_values, dim=0)
        dist = self.choose_distribution(all_values=all_values, player=game.current_player)
        # take the sum of all values, weighted by the distribution
        values = dist.view((1, -1))@all_values
        return dist, values.flatten()

    def get_policy_value(self, game: SelectionGame, selection_moves=None, special_moves=None):
        """
        averages runs of self.minimax_search to get optimal policy approximation
            if self.iterations is 1, this is equivalent to self.minimax_search
        """
        moves = ((list(selection_moves) if selection_moves is not None else []) +
                 (list(special_moves) if special_moves is not None else []))
        if not moves:
            moves = None

        dist, values = self.minimax_search(game=game, moves=moves)
        for _ in range(self.iterations - 1):
            dp, vp = self.minimax_search(game=game, moves=moves)
            dist += dp
            values += vp
        dist, values = dist/self.iterations, values/self.iterations
        return dist, values


if __name__ == '__main__':
    from aleph0.examples.tictactoe import Toe

    # if run on initial game, takes a while, then returns that every move is a tying move
    # distribution is uniform over all moves, and value is (.5,.5)
    game = Toe()

    # after taking any move, the result is the same, but runs about 9 times faster
    # distribution is uniform over the four non-losing moves, and value is (.5,.5)
    game = game.make_move(((0, 1),))

    # this is a losing move for player 1.
    # if run on this game, algorithm quickly terminates to return the one winning move for player 0
    game = game.make_move(((2, 2),))

    print(game)
    ex = Exhasutive()
    print(list(game.get_all_valid_moves()))
    print(ex.get_policy_value(game))
