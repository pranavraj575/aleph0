import os, shutil, torch, pickle
from torch import nn

from aleph0.game import FixedSizeSubsetGame
from aleph0.algs.algorithm import Algorithm
from aleph0.networks import FFN
from collections import deque


class DQNFFN(nn.Module):
    def __init__(self,
                 num_actions,
                 obs_shape,
                 underlying_set_shape,
                 output_dim,
                 action_embedding_dim=32,
                 piece_embedding=None,
                 piece_embedding_dim=32,
                 num_pieces=None,
                 hidden_layers=(64, 64),
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
            for item in underlying_set_shape:
                piece_input_dim = int(piece_input_dim*item)
            num_piece_dims = len(underlying_set_shape)
            if num_piece_dims == 0:
                self.piece_embed = nn.Embedding(num_embeddings=num_pieces,
                                                embedding_dim=piece_embedding_dim,
                                                )
            else:
                self.flatten_piec = nn.Flatten(-num_piece_dims, -1)
                self.piece_embed = nn.Linear(in_features=piece_input_dim,
                                             out_features=piece_embedding_dim,
                                             )
        else:
            self.piece_embed = piece_embedding
        self.board_flat = nn.Flatten(1, -1)
        self.action_embed = nn.Embedding(num_embeddings=num_actions,
                                         embedding_dim=action_embedding_dim,
                                         )
        self.ffn = FFN(output_dim=output_dim,
                       hidden_layers=hidden_layers,
                       input_dim=piece_embedding_dim*board_dim + list(input_vec_size)[0] + action_embedding_dim,
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
        output = self.ffn.forward(sa)
        return output


class DQNAlg(Algorithm):
    def __init__(self,
                 num_actions,
                 obs_shape,
                 underlying_set_shape,
                 num_players,
                 underlying_set_size=None,
                 gamma=.99,
                 softmax_constant=10.,
                 ):
        """
        Args:
            num_actions: number of possible actions
            obs_shape: fixed shape of observation from game
            underlying_set_shape: shape of underlying set (discrete sets must be shape ())
            num_players: number of players
            underlying_set_size: number of possible elements of underlying_set
                must specify if underly
            gamma: to use for q calculations
            softmax_constant: policy is obtained by softmax(values*softmax_constant)
                default 10 to make 0 and 1 values make sense
        """
        super().__init__()
        if underlying_set_shape == ():
            assert underlying_set_size is not None;
            "if underlying set is discrete, must be finite size"
        self.dqn = DQNFFN(num_actions=num_actions,
                          obs_shape=obs_shape,
                          underlying_set_shape=underlying_set_shape,
                          num_pieces=underlying_set_size,
                          output_dim=num_players,
                          )
        self.optim = torch.optim.Adam(self.dqn.parameters())
        self.gamma = gamma
        self.buffer = deque(maxlen=1000)
        self.softmax_constant = softmax_constant

    def save(self, save_dir):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        super().save(save_dir=save_dir)
        dic = {
            'model': self.dqn.state_dict(),
            'optim': self.optim.state_dict(),
        }
        torch.save(dic, os.path.join(save_dir, 'model_stuff.pkl'))
        f = open(os.path.join(save_dir, 'buffer.pkl'), 'wb')
        pickle.dump(self.buffer, f)
        f.close()

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        dic = torch.load(os.path.join(save_dir, 'model_stuff.pkl'),
                         weights_only=True)
        self.dqn.load_state_dict(dic['model'])
        self.optim.load_state_dict(dic['optim'])
        f = open(os.path.join(save_dir, 'buffer.pkl'), 'rb')
        self.buffer = pickle.load(f)
        f.close()

    def add_to_buffer(self,
                      game: FixedSizeSubsetGame,
                      move,
                      next_game: FixedSizeSubsetGame,
                      ):
        if next_game.is_terminal():
            result = torch.tensor(next_game.get_result())
        else:
            result = self.get_value(game=next_game).detach()
        # technically current_reward + gamma*future reward
        # however, current reward is 0 always
        target = self.gamma*result

        # invert the permuation
        # normally, the output of model assumes current player is player 0 (to make it easier)
        # permutation_to_standard_pos encodes where the correct value of each player is
        # i.e. true_values[:]=values[permutation_to_standard_pos]
        # however, we want to learn permuted values
        # thus, we invert this: fake_target[permutation_to_standard_pos]=target[:]

        target[game.permutation_to_standard_pos] = target.clone()

        self.buffer.append((
            game.batch_obs,
            torch.tensor([game.move_to_idx(move)]),
            target.view((1, -1))
        )
        )

    def sample_buffer(self, batch_size):
        obs, moves, targets = [[], [], []], [], []
        for i in torch.randint(0, len(self.buffer), (batch_size,)):
            o, m, t = self.buffer[i]
            for tl, to in zip(obs, o):
                tl.append(to)
            moves.append(m)
            targets.append(t)

        return tuple(torch.cat(t, dim=0) for t in obs), torch.cat(moves, dim=0), torch.cat(targets, dim=0)

    def learn_from_buff(self, batch_size):
        obs, moves, targets = self.sample_buffer(batch_size=batch_size)
        self.optim.zero_grad()
        output = self.dqn.forward(obs=obs, action=moves)
        criterion = nn.MSELoss()
        loss = criterion.forward(output, targets.view(output.shape))
        loss.backward()
        self.optim.step()
        return loss.item()

    def train_episode(self, game: FixedSizeSubsetGame, batch_size=128, epsilon=.05, depth=float('inf')):
        """
        trains a full episode of game, samples buffer at end
        Args:
            game: starting position
            batch_size: number of elements to sample at end of game (if 0, do not train at end)
            epsilon: randomness to use as exploration
            depth: max depth to go to
        Returns:
            loss
        """
        while depth > 0 and not game.is_terminal():
            moves = list(game.get_all_valid_moves())
            pol, val = self.get_policy_value(game, moves=moves)
            # add noise
            pol = (1 - epsilon)*pol + epsilon/len(pol)
            move = moves[torch.multinomial(pol, 1).item()]
            next_game = game.make_move(move)
            self.add_to_buffer(game, move, next_game)
            game = next_game
            depth -= 1
        if batch_size > 0:
            return self.learn_from_buff(batch_size=batch_size)
        else:
            return None

    def get_q_values(self, game, moves=None):
        """
        returns q values of game (permutes them so order is correct)
        Args:
            game:
            moves:
        """
        if moves is None:
            moves = list(game.get_all_valid_moves())
        batch_obs = game.batch_obs
        batch_obs = tuple(torch.cat([item for _ in moves], dim=0) for item in batch_obs)
        values = self.dqn.forward(obs=batch_obs, action=torch.tensor([game.move_to_idx(move) for move in moves]))
        permuation = game.permutation_to_standard_pos

        # permute them
        # values are fake values, assuming current player is player 0
        # permutation_to_standard_pos encodes where each player was sent
        return values[:, permuation]

    def get_value(self, game: FixedSizeSubsetGame, moves=None):
        """
        gets values of game for all players (permuted correctly)
        checks all q vlaues, then maximizes for game.current_player
        uses the q values at that move for all players
        """
        if game.is_terminal():
            return game.get_result()[game.current_player]
        values = self.get_q_values(game=game, moves=moves)
        return max([value for value in values], key=lambda v: v[game.current_player])

    def get_policy_value(self, game: FixedSizeSubsetGame, moves=None):
        move_values = self.get_q_values(game=game, moves=moves)
        pol = torch.softmax(move_values[:, game.current_player]*self.softmax_constant, dim=-1)
        values = torch.zeros(game.num_players)
        for prob, val in zip(pol, move_values):
            values += val*prob
        return pol.detach(), values.detach()


def DQNAlg_from_game(game: FixedSizeSubsetGame, gamma=.99, softmax_constant=10.):
    return DQNAlg(num_actions=game.possible_move_cnt(),
                  obs_shape=game.fixed_obs_shape(),
                  underlying_set_shape=game.get_underlying_set_shape(),
                  num_players=game.num_players,
                  underlying_set_size=game.underlying_set_size(),
                  gamma=gamma,
                  softmax_constant=softmax_constant,
                  )


if __name__ == '__main__':
    from aleph0.examples.tictactoe import Toe

    torch.random.manual_seed(1)

    game = Toe()
    alg = DQNAlg_from_game(game=game)
    game = game.make_move(((0, 1),))
    game = game.make_move(((2, 2),))
    game = game.make_move(((0, 2),))
    game = game.make_move(((0, 0),))
    game = game.make_move(((1, 1),))
    game = game.make_move(((2, 0),))
    print(game)
    print(torch.round(alg.get_q_values(game=game), decimals=2))

    # should learn the correct winning move
    # in this game, player 0 has one winning move and two losing moves
    # thus, algorithm should result in q values of [(0,1),(0,1),(1,0)]
    for i in range(500):
        alg.train_episode(game=game, epsilon=.1)
    print(torch.round(alg.get_q_values(game=game), decimals=2))

    # now initializing new algorithm (potentially from save point) where we train starting from empty board
    # ideally this algorithm will learn the same thing (slower though, since it must train on the entire game)
    # also this trining will take 9 times longer since games are 9 times as long
    alg = DQNAlg_from_game(game=game)
    if os.path.exists('test'):
        alg.load('test')
        print('loaded value')
        print(torch.round(alg.get_q_values(game=game), decimals=2))

    for i in range(500):
        alg.train_episode(game=Toe(), epsilon=.1)
        print(i, end='         \r')
    alg.save('test')
    print(torch.round(alg.get_q_values(game=game), decimals=2))
