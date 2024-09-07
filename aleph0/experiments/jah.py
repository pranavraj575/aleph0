import math, argparse, torch
from aleph0.experiments.common.arg_stuff import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=None
# for some reason this is faster, probably because we are parallelizing each batch
# (parallelinzing that would be annoying)
print('using device', device)
PARSER = argparse.ArgumentParser()

add_experiment_args(parse=PARSER,
                    ident='toe_test',
                    default_epochs=1500,
                    default_test_games=50,
                    default_ckpt_freq=25,
                    default_test_freq=1,
                    )
add_trans_args(parse=PARSER,
               default_dim_feedforward=64,
               default_dropout=.1,
               default_embed_dim=64,
               default_num_heads=4,
               default_num_layers=3,
               )
add_aleph_args(parse=PARSER,
               default_num_reads=420,
               default_buffer_capacity=10000,
               default_batch_size=512,
               default_minibatch_size=256,
               )

PARSER.add_argument('--game-state-search', action='store_true', required=False,
                    help="check all non-terminal tic tac toe boards to see if the policy obtained is correct")
args = PARSER.parse_args()
import torch, numpy as np, random, os, sys, time

from aleph0.examples.tictactoe import Toe
from aleph0.algs import Human, Exhasutive, Randy, AlephZero, play_game
from aleph0.networks.architect import AutoTransArchitect
from aleph0.networks.buffers import ReplayBufferDiskStorage

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

game = Toe()

torch.random.manual_seed(1)
np.random.seed(1)
random.seed(1)

ident = args.ident + get_trans_ident(args=args) + get_aleph_ident(args=args)
print('ident:', ident)

alg = AlephZero(network=AutoTransArchitect(sequence_dim=game.sequence_dim,
                                           selection_size=game.selection_size,
                                           additional_vector_dim=game.get_obs_vector_shape(),
                                           underlying_set_shapes=game.get_underlying_set_shapes(),
                                           underlying_set_sizes=game.underlying_set_sizes(),
                                           special_moves=game.special_moves,
                                           num_players=game.num_players,
                                           encoding_nums=(10, 10),
                                           base_periods_pre_exp=[-math.log(2), -math.log(2)],
                                           embedding_dim=args.embed_dim,
                                           dim_feedforward=args.dim_feedforward,
                                           dropout=args.dropout,
                                           num_board_layers=args.num_layers,
                                           nhead=args.num_heads,
                                           device=device,
                                           ),
                replay_buffer=ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, 'data', 'temp', ident),
                                                      capacity=args.buffer_capacity,
                                                      device=device,
                                                      ),
                GameClass=Toe,
                default_num_reads=args.num_reads,
                )

save_dir = os.path.join(DIR, 'data', ident)
plt_dir = os.path.join(save_dir, 'plot')

testing_trial_names = [
    'random',
    'exhaustive',
]
name_map = {'random': 'Random',
            'exhaustive': 'Optimal',
            }


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
exhaust = Exhasutive(cache_depth_rng=(0, float('inf')))
testing_agents = [[Randy()],
                  [exhaust],
                  ]

if args.play:
    print(play_game(Toe(), [Human(), alg], print_dist=True)[0])
    print(play_game(Toe(), [alg, Human()], print_dist=True)[0])
if args.game_state_search:
    import itertools

    shape = Toe().get_obs_board_shape()
    i = 0
    skipped = 0
    fail_mle = 0
    fail_corr = 0
    fail = 0
    for choice in itertools.product((Toe.EMPTY, Toe.P0, Toe.P1),
                                    repeat=int(torch.prod(torch.tensor(shape)))):

        pee0 = choice.count(Toe.P0)
        pee1 = choice.count(Toe.P1)
        if pee0 >= pee1 and pee1 >= pee0 - 1:
            board = torch.tensor(choice).view(shape)
            game = Toe(board=board,
                       current_player=Toe.P0 if pee0 == pee1 else Toe.P1,
                       )
            if not game.is_terminal():
                true_p, true_v = exhaust.get_policy_value(game=game)
                pred_p, pred_v = alg.get_policy_value(game=game)
                optimal_p = (true_p > 0)
                if torch.all(optimal_p):
                    skipped += 1
                    continue  # we do not care about these

                mle_p = torch.zeros_like(pred_p)
                mle_p[torch.argmax(pred_p)] = 1

                # probability of picking an optimal move
                correlation = torch.sum(pred_p*optimal_p)
                # greater than 0 if the 'max' policy picked an optimal move
                max_corr = torch.sum(mle_p*optimal_p)
                # if we are less than 50% choosing the best move or if the MLE choice is not optimal
                if (correlation < .5) or (max_corr == 0):
                    fail += 1
                    print()
                    if (correlation < .5):
                        fail_corr += 1
                        print('probility of choosing optimal move:', correlation.item())
                    if max_corr == 0:
                        fail_mle += 1
                        print('mle move choice:', torch.argmax(pred_p).item())
                    print(game)
                    print('policy')
                    print('true', true_p.numpy())
                    print('pred', pred_p.numpy())
                    print('value')
                    print('true', true_v.numpy())
                    print('pred', pred_v.numpy())
                i += 1
    print('total games', i)
    print('skipped games (no bad moves)', skipped)
    print('total fails', fail)
    print('total mle fails', fail_mle)
    print('total corr fails', fail_corr)
    quit()
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
