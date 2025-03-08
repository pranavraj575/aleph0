import numpy as np
import os
import itertools


def smooth(arr, n):
    """
    smooth an array by taking the average over the previous/next n values
    Args:
        arr: np array
        n: smoothing radius, n>=1
    """
    out = np.zeros_like(arr)
    out += arr
    for i in range(1, 1 + n):
        out[i:] += arr[:-i]
        out[:i] += arr[:i]  # duplicate these

        out[:-i] += arr[i:]
        out[-i:] += arr[-i:]  # duplicate these

    return out/(2*n + 1)


def plot_training_curves(plt_dir, epoch_infos, testing_trial_names, name_map, smooth_radius=1):
    """
    plot a bunch of stuff about the trial losses and such
    Args:
        plt_dir: directory to save plots in, creates this if not existing
        epoch_infos: AlephZero.epoch_infos, a list of dictionaries
        testing_trial_names: list of agents that we tested against (i.e. ['random','mcts'])
        name_map: map from testing_trial_names to real name of each agent for use in titles ({'random':'Random','mcts':'MCTS'})
        smooth_radius: radius for smoothing graphs
    """
    from matplotlib import pyplot as plt

    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    print('plotting')
    tested_epoch_infos = [epoch_info for epoch_info in epoch_infos if 'testing' in epoch_info]
    tested_x = [epoch_info['epoch'] for epoch_info in tested_epoch_infos]

    def cnvrt_to_int_key(tup):
        """
        gets rid of annoying tensor tuples for permutations like [tensor(2), tensor(0), tensor(1)]
        """
        if type(tup[0]) != int:
            return tuple(int(p) for p in tup)
        else:
            return tuple(tup)

    for smoo, split_by_order in itertools.product((False, True), repeat=2):
        prefix = ('smooth_' if smoo else '')
        just_win_rates = dict()
        just_tie_rates = dict()
        just_loss_rates = dict()
        all_all_perms = set()
        for trial_name in testing_trial_names:
            # get all used orders of agents, where '0' is self
            all_perms = set().union(
                *({cnvrt_to_int_key(test['perm']) for test in epoch_info['testing'][trial_name]}
                  for epoch_info in tested_epoch_infos)
            )
            all_all_perms = all_all_perms.union(all_perms)
            # plot win/tie/loss rates against each testing agent
            if split_by_order:
                check = [(perm,) for perm in all_perms]
            else:
                check = [all_perms]
            for perms in check:
                # perms is a list of perms to check
                # is either a singleton list of one permutaiton or the list of all_perms
                if split_by_order:
                    ourder = str(perms[0].index(0))
                else:
                    ourder = None

                self_outcomes = [
                    [
                        item['self_outcome'] for item in epoch_info['testing'][trial_name]
                        if cnvrt_to_int_key(item['perm']) in perms
                    ]
                    for epoch_info in tested_epoch_infos
                ]
                losses, ties, wins = [np.array([t.count(val)/len(t) for t in self_outcomes]) for val in (0., .5, 1.)]
                if split_by_order:
                    k = trial_name, perms
                else:
                    k = trial_name
                just_win_rates[k] = wins
                just_loss_rates[k] = losses
                just_tie_rates[k] = ties
                if smoo:
                    losses = smooth(losses, n=smooth_radius)
                    ties = smooth(ties, n=smooth_radius)
                    wins = smooth(wins, n=smooth_radius)
                plt.fill_between(x=tested_x, y1=0, y2=losses, color='red', label='losses', alpha=.69)
                plt.fill_between(x=tested_x, y1=losses, y2=losses + ties, color='purple', label='ties', alpha=.68)
                plt.fill_between(x=tested_x, y1=losses + ties, y2=1, color='blue', label='wins', alpha=.68)

                plt.title('performance against ' +
                          str(name_map[trial_name]) +
                          (' (smoothing radius ' + str(smooth_radius) + ')' if smoo else ''))
                plt.legend()
                plt.ylim((0, 1))
                plt.xlabel('epochs')
                plt.ylabel('win/tie/loss rates')
                fn = os.path.join(plt_dir,
                                  prefix +
                                  ('player_' + ourder + '_' if split_by_order else '') +
                                  'game_dist_against_' + name_map[trial_name] +
                                  '.png'
                                  )
                plt.savefig(fn)
                print('saved', fn)

                plt.close()

        for pltname, guy in (('win', just_win_rates),
                             ('tie', just_tie_rates),
                             ('loss', just_loss_rates),
                             ):

            if split_by_order:
                check = [(perm,) for perm in all_all_perms]
            else:
                check = [all_all_perms]

            # plot win/tie/loss rates against all testing agents
            for perms in check:
                if split_by_order:
                    ourder = str(perms[0].index(0))
                else:
                    ourder = None

                for trial_name in testing_trial_names:
                    if split_by_order:
                        k = trial_name, perms
                    else:
                        k = trial_name
                    thing = guy[k]
                    if smoo:
                        thing = smooth(thing, n=smooth_radius)
                    plt.plot(tested_x, thing, label=name_map[trial_name])
                plt.xlabel('epochs')
                plt.legend()
                plt.title(pltname + ' rates' + (' (smoothing radius ' + str(smooth_radius) + ')' if smoo else ''))
                plt.ylim((0, 1))
                fn = os.path.join(plt_dir, prefix +
                                  ('player_' + ourder + '_' if split_by_order else '') +
                                  'all_' +
                                  pltname +
                                  '_rates.png'
                                  )
                plt.savefig(fn)
                print('saved', fn)
                plt.close()
    # plot the losses
    all_x = [epoch_info['epoch'] for epoch_info in epoch_infos]
    for combine, log in itertools.product((True, False), repeat=2):
        for key, label in (('overall_loss', 'Overall'),
                           ('policy_loss', 'Policy'),
                           ('value_loss', 'Value'),
                           ):
            losses = np.array([ei[key] for ei in epoch_infos])
            if log:
                losses = np.log(losses)
            plt.plot(all_x, losses, label=label)
            if not combine:
                fn = os.path.join(plt_dir, ('log_' if log else '') + 'losses_' + label + '.png')
                plt.xlabel('epochs')
                plt.ylabel(('log ' if log else '') + 'loss')
                plt.title(label)
                plt.savefig(fn)
                print('saved', fn)
                plt.close()
        if combine:
            plt.legend()
            fn = os.path.join(plt_dir, ('log_' if log else '') + 'losses.png')
            plt.xlabel('epochs')
            plt.ylabel(('log ' if log else '') + 'loss')
            plt.savefig(fn)
            print('saved', fn)
            plt.close()
    print('done plotting')
