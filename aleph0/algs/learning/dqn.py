import torch
from torch import nn

from aleph0.game import FixedSizeSubsetGame
from aleph0.algs.algorithm import Algorithm
from aleph0.networks import FFN


class DQNFFN(nn.Module):
    def __init__(self,
                 num_actions,
                 obs_shape,
                 piece_shape,
                 action_embedding_dim=32,
                 piece_embedding=None,
                 piece_embedding_dim=32,
                 num_pieces=None,
                 hidden_layers=[64, 64]
                 ):
        super().__init__()
        board_shape, _, input_vec_size = obs_shape
        board_dim = 1
        for item in board_shape:
            board_dim = int(board_dim*item)
        self.flatten_piec = None
        if piece_embedding is None:
            self.piece_embed = nn.Identity()
            piece_input_dim = 1
            for item in piece_shape:
                piece_input_dim = int(piece_input_dim*item)
            num_piece_dims = len(piece_shape)
            if num_piece_dims == 0:
                self.piece_embed = nn.Embedding(num_embeddings=num_pieces,
                                                embedding_dim=piece_embedding_dim,
                                                )
            else:
                self.flatten_piec = nn.Flatten(-num_piece_dims, -1)
                self.piece_embed = nn.Linear(in_features=piece_input_dim, out_features=piece_embedding_dim)
        else:
            self.piece_embed = piece_embedding
        self.board_flat = nn.Flatten(1, -1)
        self.action_embed = nn.Embedding(num_embeddings=num_actions,
                                         embedding_dim=action_embedding_dim,
                                         )
        self.ffn = FFN(output_dim=1,
                       hidden_layers=hidden_layers,
                       input_dim=piece_embedding_dim*board_dim + input_vec_size + action_embedding_dim
                       )

    def forward(self, obs, action):
        """
        obs is an (N,*) board, positions, and a (N,T) batch of vectors
        aciton s an (N,) batch of actions
        """
        board, _, vec = obs
        if self.flatten_piec is not None:
            board = self.flatten_piec(board)
        board = self.piece_embed.forward(board)
        board = self.board_flat.forward(board)
        sa = torch.cat((board, vec, self.action_embed(action)), dim=1)
        return self.ffn.forward(sa)


class DQNAlg(Algorithm):
    def __init__(self, game: FixedSizeSubsetGame):
        self.dqn = DQNFFN(num_actions=game.possible_move_cnt(),
                          obs_shape=game.observation_shape,
                          piece_shape=game.get_obs_piece_shape(),
                          num_pieces=game.num_pieces() if game.get_obs_piece_shape() == () else None,
                          )
        self.move_to_idx = {game.index_to_move(i): i for i in range(game.possible_move_cnt())}

    def get_policy_value(self, game: FixedSizeSubsetGame, moves=None):
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
        if moves is None:
            moves = list(game.get_all_valid_moves())
        batch_obs = game.batch_obs

        batch_obs = tuple(torch.cat([item for _ in moves], dim=0) for item in batch_obs)
        values = self.dqn.forward(obs=batch_obs, action=torch.tensor([self.move_to_idx[move] for move in moves]))
        return values.flatten()


if __name__ == '__main__':
    from aleph0.examples.tictactoe import Toe

    alg = DQNAlg(Toe())
    print(alg)
    print(alg.dqn.forward(obs=Toe().batch_obs, action=torch.tensor([0])))
    print(alg.get_policy_value(Toe()))
