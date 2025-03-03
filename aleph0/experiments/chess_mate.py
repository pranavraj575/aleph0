import argparse, torch, os, math, sys, time
import numpy as np
from aleph0.experiments.common.plotting_anal import plot_training_curves

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--material', action='store', required=False, default='qr',
                    help='what material row to use, q queen, r rook (Default is qr)')
PARSER.add_argument('--ident', action='store', required=False, default='mating_test',
                    help="folder name")
PARSER.add_argument('--epochs', type=int, required=False, default=4000,
                    help="number of epochs")
PARSER.add_argument('--embed-dim', type=int, required=False, default=128,
                    help="embed dim")
PARSER.add_argument('--num-layers', type=int, required=False, default=4,
                    help="num layers")
PARSER.add_argument('--dropout', type=float, required=False, default=0.1,
                    help="dropout")
PARSER.add_argument('--play', action='store_true', required=False,
                    help="whether to play game with alg")
PARSER.add_argument('--reset', action='store_true', required=False,
                    help="reset training")
PARSER.add_argument('--plot', action='store_true', required=False,
                    help="whether to plot")
PARSER.add_argument('--smooth-radius', type=int, required=False, default=10,
                    help="radius of smoothing window of plot")
args = PARSER.parse_args()
from aleph0.examples.chess import Chess2d
from aleph0.examples.chess.game import P, Board

material = args.material.lower()


def create_game():
    board = torch.zeros(Board.BOARD_SHAPE, dtype=torch.long)
    I, J = Board.BOARD_SHAPE
    board[I - 1, J - 1] = P.as_player(P.KING, P.P1)
    board[0, 0] = P.as_player(P.KING, P.P0)
    mapping = {'q': P.QUEEN,
               'r': P.ROOK,
               }
    for p in material:
        piece = P.as_player(mapping[p], P.P0)
        a, b = 0, 0
        # dont let the game start out with king in check
        while board[a, b] != P.EMPTY or (a >= I - 2 and b >= J - 2) or (a == b):
            a, b = torch.randint(0, I - 1, (1,)).item(), torch.randint(0, J - 1, (1,)).item()

        board[a, b] = piece
    game = Chess2d(initial_board=Board(board=board))
    return game


test_game = create_game()
print(test_game)

from aleph0.algs import Human, MCTS, Randy, AlephZero, play_game
from aleph0.networks.architect import AutoTransArchitect
from aleph0.networks.buffers import ReplayBufferDiskStorage

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

ident = args.ident
ident += '_material_' + material
ident += '_embed_dim_' + str(args.embed_dim)
ident += '_lyrs_' + str(args.num_layers)
ident += '_drop_' + str(float(args.dropout))

alg = AlephZero(network=AutoTransArchitect(sequence_dim=test_game.sequence_dim,
                                           selection_size=test_game.selection_size,
                                           additional_vector_dim=test_game.get_obs_vector_shape(),
                                           underlying_set_shapes=test_game.get_underlying_set_shapes(),
                                           underlying_set_sizes=test_game.underlying_set_sizes(),
                                           special_moves=test_game.special_moves,
                                           num_players=test_game.num_players,
                                           encoding_nums=(10, 10),
                                           base_periods_pre_exp=[-math.log(2), -math.log(2)],
                                           embedding_dim=args.embed_dim,
                                           dim_feedforward=64,
                                           dropout=args.dropout,
                                           num_board_layers=args.num_layers,
                                           nhead=4,
                                           ),
                replay_buffer=ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, 'data', 'temp', ident),
                                                      capacity=100000,
                                                      ),
                GameClass=Chess2d,
                default_num_reads=690,
                )

save_dir = os.path.join(DIR, 'data', ident)
plt_dir = os.path.join(save_dir, 'plot')

testing_agents = [[Randy()],
                  [alg],
                  # [MCTS(num_reads=420)]
                  ]
testing_trial_names = [
    'random',
    'self',
    'mcts',
]
name_map = {'random': 'Random',
            'self': 'Self',
            'mcts': 'MCTS',
            }

batch = 512
mini = 256

if not args.reset and os.path.exists(save_dir):
    print('loading algorithm from', save_dir)
    alg.load(save_dir)
    print('epochs pretrained:', alg.epochs)
    if args.plot:
        plot_training_curves(plt_dir=plt_dir,
                             epoch_infos=alg.epoch_infos,
                             testing_trial_names=testing_trial_names,
                             name_map=name_map,
                             smooth_radius=args.smooth_radius,
                             )
if args.play:
    print(play_game(test_game, [alg, Human()], print_dist=True)[0])
    print(play_game(test_game, [Human(), alg], print_dist=True)[0])
while alg.info['epochs'] < args.epochs:
    these_testing_agents = None
    start = time.time()
    if not alg.info['epochs']%1:
        these_testing_agents = testing_agents
    alg.epoch(game=create_game(),
              batch_size=batch,
              minibatch_size=mini,
              testing_agents=these_testing_agents,
              testing_trial_names=testing_trial_names,
              num_test_games=20,
              depth=50,
              testing_possible_perms=[
                  [[0, 1], ] for _ in testing_agents
              ]
              )
    print(alg.epochs, 'time', round(time.time() - start), '         ')

    if not alg.info['epochs']%10:
        print('saving')
        alg.save(save_dir)
        print('done saving')

# clear buffer
alg.clear()
