import numpy as np
import os


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
    just_win_rates = dict()
    just_tie_rates = dict()
    just_loss_rates = dict()
    tested_epoch_infos = [epoch_info for epoch_info in epoch_infos if 'testing' in epoch_info]
    tested_x = [epoch_info['epoch'] for epoch_info in tested_epoch_infos]
    for trial_name in testing_trial_names:
        # plot win/tie/loss rates against each testing agents
        for smoo in (True, False):
            self_outcomes = [[item['self_outcome'] for item in epoch_info['testing'][trial_name]]
                             for epoch_info in tested_epoch_infos]
            losses, ties, wins = [np.array([t.count(val)/len(t) for t in self_outcomes]) for val in (0., .5, 1.)]
            just_win_rates[trial_name] = wins
            just_loss_rates[trial_name] = losses
            just_tie_rates[trial_name] = ties
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
                              ('smooth_' if smoo else '') + 'game_dist_against_' + name_map[trial_name] + '.png')
            plt.savefig(fn)
            print('saved', fn)

            plt.close()
    for pltname, guy in (('win', just_win_rates),
                         ('tie', just_tie_rates),
                         ('loss', just_loss_rates),
                         ):
        # plot win/tie/loss rates against all testing agents
        for smoo in (True, False):
            for trial_name in guy:
                thing = guy[trial_name]
                if smoo:
                    thing = smooth(thing, n=smooth_radius)
                plt.plot(tested_x, thing, label=name_map[trial_name])
            plt.xlabel('epochs')
            plt.legend()
            plt.title(pltname + ' rates' + (' (smoothing radius ' + str(smooth_radius) + ')' if smoo else ''))
            plt.ylim((0, 1))
            fn = os.path.join(plt_dir, ('smooth_' if smoo else '') + 'all_' + pltname + '_rates.png')
            plt.savefig(fn)
            print('saved', fn)
            plt.close()
    # plot the losses
    all_x = [epoch_info['epoch'] for epoch_info in epoch_infos]
    for combine in (True, False):
        for key, label in (('overall_loss', 'Overall'),
                           ('policy_loss', 'Policy'),
                           ('value_loss', 'Value'),
                           ):
            plt.plot(all_x, [ei[key] for ei in epoch_infos], label=label)
            if not combine:
                fn = os.path.join(plt_dir, 'losses_' + label + '.png')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.title(label)
                plt.savefig(fn)
                print('saved', fn)
                plt.close()
        if combine:
            plt.legend()
            fn = os.path.join(plt_dir, 'losses.png')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.savefig(fn)
            print('saved', fn)
            plt.close()
    print('done plotting')
