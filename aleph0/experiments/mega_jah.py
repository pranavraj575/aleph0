import math, argparse

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--plot', action='store_true', required=False,
                    help="whether to plot")
PARSER.add_argument('--smooth-radius', type=int, required=False, default=10,
                    help="radius of smoothing window of plot")
PARSER.add_argument('--play', action='store_true', required=False,
                    help="whether to play game with alg")
PARSER.add_argument('--reset', action='store_true', required=False,
                    help="reset training")

PARSER.add_argument('--ident', action='store', required=False, default='ult_toe_test',
                    help="folder name")
PARSER.add_argument('--epochs', type=int, required=False, default=2000,
                    help="number of epochs")
PARSER.add_argument('--embed-dim', type=int, required=False, default=128,
                    help="embed dim")
PARSER.add_argument('--num-layers', type=int, required=False, default=4,
                    help="num layers")
PARSER.add_argument('--dropout', type=float, required=False, default=0.1,
                    help="dropout")
args = PARSER.parse_args()
import torch, numpy as np, random, os, sys, time

from aleph0.examples.tictactoe import UltimateToe
from aleph0.algs import Human, Randy, AlephZero, play_game
from aleph0.networks.architect import AutoTransArchitect
from aleph0.networks.buffers import ReplayBufferDiskStorage

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

game = UltimateToe()

torch.random.manual_seed(1)
np.random.seed(1)
random.seed(1)

ident = args.ident
ident += '_embed_dim_' + str(args.embed_dim)
ident += '_lyrs_' + str(args.num_layers)
ident += '_drop_' + str(float(args.dropout))

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
                                           dim_feedforward=64,
                                           dropout=args.dropout,
                                           num_board_layers=args.num_layers,
                                           nhead=4,
                                           ),
                replay_buffer=ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, 'data', 'temp', ident),
                                                      capacity=100000,
                                                      ),
                GameClass=UltimateToe,
                default_num_reads=1069,
                )

save_dir = os.path.join(DIR, 'data', ident)
plt_dir = os.path.join(save_dir, 'plot')

testing_trial_names = [
    'random',
    'self',
]
name_map = {'random': 'Random',
            'self': 'Self',
            }

batch = 512
mini = 256


def smooth(arr, n):
    out = np.zeros_like(arr)
    out += arr
    for i in range(1, 1 + n):
        out[i:] += arr[:-i]
        out[:i] += arr[:i]  # duplicate these

        out[:-i] += arr[i:]
        out[-i:] += arr[-i:]  # duplicate these

    return out/(2*n + 1)


if not args.reset and os.path.exists(save_dir):
    print('loading algorithm from', save_dir)
    alg.load(save_dir)
    print('epochs pretrained:', alg.epochs)
    if args.plot:
        from matplotlib import pyplot as plt

        if not os.path.exists(plt_dir):
            os.makedirs(plt_dir)
        print('plotting')
        just_win_rates = dict()
        just_tie_rates = dict()
        just_loss_rates = dict()
        epoch_infos = alg.epoch_infos
        x = self_outcomes = [epoch_info['epoch'] for epoch_info in epoch_infos]
        for trial_name in testing_trial_names:
            for smoo in (True, False):
                self_outcomes = [[item['self_outcome'] for item in epoch_info['testing'][trial_name]]
                                 for epoch_info in epoch_infos]
                losses, ties, wins = [np.array([t.count(val)/len(t) for t in self_outcomes]) for val in (0., .5, 1.)]
                just_win_rates[trial_name] = wins
                just_loss_rates[trial_name] = losses
                just_tie_rates[trial_name] = ties
                if smoo:
                    losses = smooth(losses, n=args.smooth_radius)
                    ties = smooth(ties, n=args.smooth_radius)
                    wins = smooth(wins, n=args.smooth_radius)
                plt.fill_between(x=x, y1=0, y2=losses, color='red', label='losses', alpha=.69)
                plt.fill_between(x=x, y1=losses, y2=losses + ties, color='purple', label='ties', alpha=.68)
                plt.fill_between(x=x, y1=losses + ties, y2=1, color='blue', label='wins', alpha=.68)

                plt.title('performance against ' +
                          str(name_map[trial_name]) +
                          (' (smoothing radius ' + str(args.smooth_radius) + ')' if smoo else ''))
                plt.legend()
                plt.ylim((0, 1))
                plt.xlabel('epochs')
                plt.ylabel('win/tie/loss rates')
                fn = os.path.join(plt_dir,
                                  ('smooth_' if smoo else '') + 'game_dist_against_' + name_map[trial_name] + '.png')
                plt.savefig(fn)
                print('saved', fn)

                plt.close()
        for pltname, guy in (('win', just_win_rates),
                             ('tie', just_tie_rates),
                             ('loss', just_loss_rates),
                             ):
            for smoo in (True, False):
                for trial_name in guy:
                    thing = guy[trial_name]
                    if smoo:
                        thing = smooth(thing, n=args.smooth_radius)
                    plt.plot(thing, label=name_map[trial_name])
                plt.xlabel('epochs')
                plt.legend()
                plt.title(pltname + ' rates' + (' (smoothing radius ' + str(args.smooth_radius) + ')' if smoo else ''))
                plt.ylim((0, 1))
                fn = os.path.join(plt_dir, ('smooth_' if smoo else '') + 'all_' + pltname + '_rates.png')
                plt.savefig(fn)
                print('saved', fn)
                plt.close()
        print('done plotting')
testing_agents = [[Randy()],
                  [alg],
                  ]

if args.play:
    print(play_game(UltimateToe(), [Human(), alg], print_dist=True)[0])
    print(play_game(UltimateToe(), [alg, Human()], print_dist=True)[0])
while alg.info['epochs'] < args.epochs:
    these_testing_agents = None
    start = time.time()
    if not alg.info['epochs']%1:
        these_testing_agents = testing_agents
    alg.epoch(game=game,
              batch_size=batch,
              minibatch_size=mini,
              testing_agents=these_testing_agents,
              testing_trial_names=testing_trial_names,
              num_test_games=100,
              )
    print(alg.epochs, 'time', round(time.time() - start), '         ')

    if not alg.info['epochs']%25:
        print('saving')
        alg.save(save_dir)
        print('done saving')

# clear buffer
alg.clear()
