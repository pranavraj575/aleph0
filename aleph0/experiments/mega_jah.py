import math, argparse
from aleph0.experiments.common.arg_stuff import *
from aleph0.experiments.common.plotting_anal import plot_training_curves

PARSER = argparse.ArgumentParser()
add_experiment_args(parse=PARSER,
                    ident='ult_toe_test',
                    default_test_games=50,
                    default_test_freq=1,
                    default_ckpt_freq=5,
                    )
add_trans_args(parse=PARSER,
               default_embed_dim=128,
               default_num_heads=4,
               default_num_layers=4,
               default_dim_feedforward=128,
               default_dropout=.1,
               )
add_aleph_args(parse=PARSER,
               default_num_reads=1069,
               default_buffer_capacity=50000,
               default_batch_size=512,
               default_minibatch_size=256,
               )
args = PARSER.parse_args()
import torch, numpy as np, random, os, sys, time

from aleph0.examples.tictactoe import UltimateToe
from aleph0.algs import Human, Randy, MCTS, AlephZero, play_game
from aleph0.networks.architect import AutoTransArchitect
from aleph0.networks.buffers import ReplayBufferDiskStorage

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

game = UltimateToe()

torch.random.manual_seed(1)
np.random.seed(1)
random.seed(1)

ident = (args.ident +
         get_trans_ident(args=args) +
         get_aleph_ident(args=args)
         )

alg = AlephZero(network=AutoTransArchitect(sequence_dim=game.sequence_dim,
                                           selection_size=game.selection_size,
                                           additional_vector_dim=game.get_obs_vector_shape(),
                                           underlying_set_shapes=game.get_underlying_set_shapes(),
                                           underlying_set_sizes=game.underlying_set_sizes(),
                                           special_moves=game.special_moves,
                                           num_players=game.num_players,
                                           encoding_nums=(10, 10, 10, 10),
                                           base_periods_pre_exp=[-math.log(2) for _ in range(4)],
                                           embedding_dim=args.embed_dim,
                                           dim_feedforward=args.dim_feedforward,
                                           dropout=args.dropout,
                                           num_board_layers=args.num_layers,
                                           nhead=args.num_heads,
                                           ),
                replay_buffer=ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, 'data', 'temp', ident),
                                                      capacity=args.buffer_capacity,
                                                      ),
                GameClass=UltimateToe,
                default_num_reads=args.num_reads,
                )

save_dir = os.path.join(DIR, 'data', ident)
plt_dir = os.path.join(save_dir, 'plot')

testing_trial_names = [
    'random',
    'self',
    # 'mcts',
]
name_map = {'random': 'Random',
            'self': 'Self',
            'mcts': 'MCTS',
            }


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
testing_agents = [[Randy()],
                  [alg],
                  # [MCTS(num_reads=args.num_reads)],
                  ]

if args.play:
    print(play_game(UltimateToe(), [Human(), alg], print_dist=True)[0])
    print(play_game(UltimateToe(), [alg, Human()], print_dist=True)[0])
while alg.info['epochs'] < args.epochs:
    these_testing_agents = None
    start = time.time()
    if not alg.info['epochs']%args.test_freq and args.num_test_games:
        these_testing_agents = testing_agents
    alg.epoch(game=game,
              batch_size=args.batch_size,
              minibatch_size=args.minibatch_size,
              testing_agents=these_testing_agents,
              testing_trial_names=testing_trial_names,
              num_test_games=args.num_test_games,
              )
    print(alg.epochs, 'time', round(time.time() - start), '         ')

    if not alg.info['epochs']%args.ckpt_freq:
        print('saving')
        alg.save(save_dir)
        print('done saving')

# clear buffer
alg.clear()
