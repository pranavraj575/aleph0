import os, shutil, torch, pickle
from torch import nn

from aleph0.game import FixedSizeSelectionGame
from aleph0.algs.algorithm import Algorithm
from aleph0.networks import FFN, PieceEmbedder, FlattenEmbedder, BoardSetEmbedder
from collections import deque


class DQNFFN(nn.Module):
    def __init__(self,
                 num_actions,
                 obs_shape,
                 underlying_set_shapes,
                 output_dim,
                 action_embedding_dim=32,
                 piece_embeddings=None,
                 piece_embedding_dims=32,
                 overall_piece_embedding_dim=32,
                 underlying_set_sizes=None,
                 hidden_layers=(64, 64),
                 ):
        super().__init__()

        board_shapes, pos_shape, input_vec_size = obs_shape
        # pos shape is (D1,...,DN, N)
        # board shape is (D1,...,DN)
        board_shape = pos_shape[:-1]
        total_board_dim = 0
        board_dims = 1
        for item in board_shape:
            board_dims = int(board_dims*item)
        total_board_dim += board_dims

        if underlying_set_sizes is None:
            underlying_set_sizes = [None for _ in board_shapes]
        if isinstance(piece_embedding_dims, int):
            piece_embedding_dims = [piece_embedding_dims for _ in board_shapes]
        if piece_embeddings is None:
            piece_embeddings = [None for _ in board_shapes]
        board_embedders = []
        for (underlying_set_shape,
             underlying_set_size,
             piece_embedding,
             piece_embedding_dim) in zip(
            underlying_set_shapes,
            underlying_set_sizes,
            piece_embeddings,
            piece_embedding_dims):
            if piece_embedding is not None:
                board_embedders.append(piece_embedding)
            else:
                if underlying_set_shape == ():
                    assert isinstance(underlying_set_size, int)
                    board_embedders.append(PieceEmbedder(embedding_dim=piece_embedding_dim,
                                                         piece_count=underlying_set_size))
                else:
                    board_embedders.append(FlattenEmbedder(input_shape=underlying_set_shape))

        self.boardsetemb = BoardSetEmbedder(final_embedding_dim=overall_piece_embedding_dim,
                                            board_embedding_list=board_embedders)
        self.flatten_board = nn.Flatten()
        self.action_embed = nn.Embedding(num_embeddings=num_actions,
                                         embedding_dim=action_embedding_dim,
                                         )
        self.ffn = FFN(output_dim=output_dim,
                       hidden_layers=hidden_layers,
                       input_dim=total_board_dim*overall_piece_embedding_dim +
                                 input_vec_size[0] +
                                 action_embedding_dim,
                       )

    def forward(self, obs, action):
        """
        obs is a list of (N,*) boards, (N,*) positions, and a (N,T) batch of vectors
        aciton s an (N,) batch of actions
        """
        boards, _, vec = obs
        # size (N, total_board_dim*overall_piece_embedding_dim)
        board_embedding = self.flatten_board(self.boardsetemb.forward(boards))

        sa = torch.cat((board_embedding, vec, self.action_embed(action)), dim=1)
        output = self.ffn.forward(sa)
        return output


class DQNAlg(Algorithm):
    def __init__(self,
                 num_actions,
                 obs_shape,
                 underlying_set_shapes,
                 num_players,
                 underlying_set_sizes=None,
                 gamma=.99,
                 softmax_constant=10.,
                 ):
        """
        Args:
            num_actions: number of possible actions
            obs_shapes: fixed shape of observations from game
            underlying_set_shapes: shapes of underlying set (discrete sets must be shape ())
            num_players: number of players
            underlying_set_size: number of possible elements of underlying_set
                must specify if underly
            gamma: to use for q calculations
            softmax_constant: policy is obtained by softmax(values*softmax_constant)
                default 10 to make 0 and 1 values make sense
        """
        super().__init__()
        self.dqn = DQNFFN(num_actions=num_actions,
                          obs_shape=obs_shape,
                          underlying_set_shapes=underlying_set_shapes,
                          underlying_set_sizes=underlying_set_sizes,
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
                      game: FixedSizeSelectionGame,
                      move,
                      next_game: FixedSizeSelectionGame,
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
        boards, poss, vecs = [], [], []
        obs, moves, targets = [boards, poss, vecs], [], []
        for i in torch.randint(0, len(self.buffer), (batch_size,)):
            o, m, t = self.buffer[i]
            bs, p, v = o
            if not boards:
                for _ in bs:
                    boards.append([])
            for bl, b in zip(boards, bs):
                bl.append(b)
            poss.append(p)
            vecs.append(v)

            moves.append(m)
            targets.append(t)
        boards = tuple(torch.cat(t, dim=0) for t in boards)
        poss = torch.cat(poss, dim=0)
        vecs = torch.cat(vecs, dim=0)
        moves = torch.cat(moves, dim=0)
        targets = torch.cat(targets, dim=0)

        return (boards, poss, vecs), moves, targets

    def learn_from_buff(self, batch_size):
        obs, moves, targets = self.sample_buffer(batch_size=batch_size)
        self.optim.zero_grad()
        output = self.dqn.forward(obs=obs, action=moves)
        criterion = nn.MSELoss()
        loss = criterion.forward(output, targets.view(output.shape))
        loss.backward()
        self.optim.step()
        return loss.item()

    def train_episode(self, game: FixedSizeSelectionGame, batch_size=128, epsilon=.05, depth=float('inf')):
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
            pol, val = self.get_policy_value(game, selection_moves=moves)
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
        batch_boards, batch_pos, batch_vec = game.batch_obs
        batch_boards = tuple(torch.cat([batch_board for _ in moves], dim=0) for batch_board in batch_boards)
        batch_pos = torch.cat([batch_pos for _ in moves], dim=0)
        batch_vec = torch.cat([batch_vec for _ in moves], dim=0)

        values = self.dqn.forward(obs=(batch_boards, batch_pos, batch_vec),
                                  action=torch.tensor([game.move_to_idx(move) for move in moves]))
        permuation = game.permutation_to_standard_pos

        # permute them
        # values are fake values, assuming current player is player 0
        # permutation_to_standard_pos encodes where each player was sent
        return values[:, permuation]

    def get_value(self, game: FixedSizeSelectionGame, moves=None):
        """
        gets values of game for all players (permuted correctly)
        checks all q vlaues, then maximizes for game.current_player
        uses the q values at that move for all players
        """
        if game.is_terminal():
            return game.get_result()[game.current_player]
        values = self.get_q_values(game=game, moves=moves)
        return max([value for value in values], key=lambda v: v[game.current_player])

    def get_policy_value(self, game: FixedSizeSelectionGame, selection_moves=None, special_moves=None):
        if selection_moves is None:
            selection_moves = list(game.get_all_valid_moves())
        if special_moves is None:
            special_moves=list(game.valid_special_moves())
        moves=selection_moves+special_moves
        move_values = self.get_q_values(game=game, moves=moves)
        pol = torch.softmax(move_values[:, game.current_player]*self.softmax_constant, dim=-1)
        values = torch.zeros(game.num_players)
        for prob, val in zip(pol, move_values):
            values += val*prob
        return pol.detach(), values.detach()


def DQNAlg_from_game(game: FixedSizeSelectionGame, gamma=.99, softmax_constant=10.):
    return DQNAlg(num_actions=game.possible_move_cnt(),
                  obs_shape=game.fixed_obs_shape(),
                  underlying_set_shapes=game.get_underlying_set_shapes(),
                  num_players=game.num_players,
                  underlying_set_sizes=game.underlying_set_sizes(),
                  gamma=gamma,
                  softmax_constant=softmax_constant,
                  )


if __name__ == '__main__':
    from aleph0.examples.tictactoe import Toe

    torch.random.manual_seed(1)

    game = Toe()
    alg = DQNAlg_from_game(game=game)
    game = game.make_move(((0, 0),))
    game = game.make_move(((1, 0),))
    game = game.make_move(((0, 1),))
    game = game.make_move(((1, 1),))

    print(game)
    print(torch.round(alg.get_q_values(game=game), decimals=2))

    # should learn the correct winning move
    # in this game, player 0 has one winning move and two losing moves
    # thus, algorithm should result in q values of [(0,1),(0,1),(1,0)]
    for i in range(10000):
        alg.train_episode(game=game, epsilon=.1)
    print(torch.round(alg.get_q_values(game=game), decimals=2))

    # now initializing new algorithm (potentially from save point) where we train starting from empty board
    # ideally this algorithm will learn the same thing (slower though, since it must train on the entire game)
    # also this trining will take 9 times longer since games are 9 times as long
    alg = DQNAlg_from_game(game=game)
    save_path = 'temp'
    if os.path.exists(save_path):
        alg.load(save_path)
        print('loaded value')
        print(torch.round(alg.get_q_values(game=game), decimals=2))
    for i in range(1000):
        alg.train_episode(game=Toe(), epsilon=.1)
        print(i, end='         \r')
    alg.save(save_path)
    print(torch.round(alg.get_q_values(game=game), decimals=2))

    from aleph0.algs import Human, play_game, Exhasutive

    outcome, _ = play_game(game=Toe(), alg_list=[Human(), alg])
    print(outcome[0])
    if outcome[0][1] == 1:
        print('you suck at this')

    for _ in range(10):
        outcome, _ = play_game(game=Toe(), alg_list=[Exhasutive(), alg])
        print(outcome)
