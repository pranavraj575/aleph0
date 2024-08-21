import os, torch, pickle
import shutil

from torch import nn

from aleph0.algs.algorithm import Algorithm
from aleph0.algs.nonlearning.mcts import UCT_search
from aleph0.algs.play_game import play_game

from aleph0.networks.buffers import ReplayBuffer
from aleph0.networks.architect.architect import Architect

from aleph0.game.selection_game import SelectionGame


class AlephZero(Algorithm):
    def __init__(self,
                 network: Architect,
                 replay_buffer: ReplayBuffer,
                 GameClass,
                 lr=.001,
                 default_num_reads=420,
                 use_mcts_in_testing=False,
                 ):
        """
        Args:
            network: Arcitect to use to go from game observation to policy/value
            replay_buffer: buffer to store (game representation, target policy, target value)
            GameClass: class of game, used to recover game from observation
            lr: lr used in optimizer
            default_num_reads: default num_reads to give when calculating MCTS
            use_mcts_in_testing: whether to use mcts when calculating final policy/value
        """
        super().__init__()

        self.network = network
        self.GameClass = GameClass
        self.optim = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = replay_buffer
        self.default_num_reads = default_num_reads
        self.info['epochs'] = 0
        self.info['epoch_infos'] = []
        self.use_mcts_in_testing = use_mcts_in_testing

    @property
    def epochs(self):
        return self.info['epochs']

    def clear(self):
        super().clear()
        self.buffer.clear()

    def save(self, save_dir):
        # save this separately from info
        epoch_infos = self.info.pop('epoch_infos')
        if epoch_infos:
            epoch_info_save_dir = os.path.join(save_dir, 'epoch_infos')
            if 'epoch_info_save_dir' in self.info and self.info['epoch_info_save_dir'] != epoch_info_save_dir:
                shutil.copytree(self.info['epoch_info_save_dir'], epoch_info_save_dir)

            if not os.path.exists(epoch_info_save_dir):
                os.makedirs(epoch_info_save_dir)

            filename = os.path.join(epoch_info_save_dir,
                                    str(epoch_infos[0]['epoch']) + '_to_' +
                                    str(epoch_infos[-1]['epoch']) + '.pkl'
                                    )
            f = open(filename, 'wb')
            pickle.dump(epoch_infos, f)
            f.close()
            self.info['epoch_info_save_dir'] = epoch_info_save_dir

        super().save(save_dir=save_dir)
        self.info['epoch_infos'] = []
        dic = {
            'model': self.network.state_dict(),
            'optim': self.optim.state_dict(),
        }
        torch.save(dic, os.path.join(save_dir, 'model.pkl'))
        self.buffer.save(save_dir=os.path.join(save_dir, 'buffer'))

    @property
    def epoch_infos(self):
        if 'epoch_info_save_dir' in self.info:
            # must grab past epoch infos
            epoch_info_save_dir = self.info['epoch_info_save_dir']
            files = [f for f in os.listdir(epoch_info_save_dir) if '_to_' in f]
            # sort by start epoch
            files.sort(key=lambda n: int(n.split('_to_')[0]))
            epoch_infos = []
            for fn in files:
                f = open(os.path.join(epoch_info_save_dir, fn), 'rb')
                epoch_infos.extend(pickle.load(f))
                f.close()
            epoch_infos.extend(self.info['epoch_infos'])
            return epoch_infos
        else:
            return self.info['epoch_infos']

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        dic = torch.load(os.path.join(save_dir, 'model.pkl'), weights_only=True)
        self.network.load_state_dict(dic['model'])
        self.optim.load_state_dict(dic['optim'])
        self.buffer.load(save_dir=os.path.join(save_dir, 'buffer'))

    def policy_loss(self, prob_dist_pred, probability_dist_targets, shift_to_zero=True):
        """
        Args:
            prob_dist_pred: (N, moves) policy prediction
            probability_dist_targets: (N, moves) policy target
            shift_to_zero: whether to shift loss output so that 0 is optimal
                this changes nothing about the gradients, it just subtracts the entropy of the targets from the loss
        Returns:

        """
        # crossentropy loss
        # annoying to use torch.nn.CrossEntropyLoss since it assumes the target is a single class as opposed to a dist
        # so just implement it directly like this
        policy_losses = -torch.sum(probability_dist_targets*torch.log(prob_dist_pred), dim=1)
        if shift_to_zero:
            # calculate the entropy of the target distribution, and subtract it
            # all this does is shift the loss so that the optimal loss is 0, achieved by copying the target dist
            # this will not affect the gradients at all, as this is a constant wrt the network params
            optimal_policy_losses = -torch.sum(probability_dist_targets*torch.log(probability_dist_targets), dim=1)
            policy_losses = policy_losses - optimal_policy_losses

        return torch.mean(policy_losses)  # average across N

    def value_loss(self, predicitons, targets):
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicitons, targets)
        return loss

    def add_to_buffer(self, game: SelectionGame, target_policy, target_values):
        target_values = target_values.reshape(1, -1)
        # increase samples by adding all symmertries of the game as well
        for sym_game, sym_policy in game.symmetries(policy_vector=target_policy):
            self.buffer.push((sym_game.representation, sym_policy.reshape(1, -1), target_values))
        # self.buffer.push((game.representation, target_policy, target_values))

    def generate_training_data(self,
                               game: SelectionGame,
                               num_reads,
                               ):
        """
        generates training data for next step of game
            training data contains PERMUTED values, as these are targets for our network
            true_values[i]=network_output[perm[i]]
        Args:
            game: non terminal game to use in UCT search
            num_reads: number of reads to send to UCT search
        Returns:
            (policy, values) obtained from UCT_search
        """
        if game.is_terminal():
            print("error, cant generate training data on a terminal game")
            return
        target_policy, true_values, root = UCT_search(game=game,
                                                      num_reads=num_reads,
                                                      policy_value_evaluator=self.policy_value_evaluator
                                                      )
        target_policy = torch.tensor(target_policy)
        true_values = torch.tensor(true_values)
        perm = game.permutation_to_standard_pos
        if perm is not None:
            # we do the inverse permutation here
            # true_values[i]=network_output[perm[i]]
            # so network_output[perm[i]]=true_values[i]
            target_values = torch.zeros_like(true_values)
            target_values[perm] = true_values
        else:
            target_values = true_values
        # convert to batch form, then push
        self.add_to_buffer(game=game,
                           target_policy=target_policy,
                           target_values=target_values,
                           )
        return target_policy, target_values, (root.next_selection_moves, root.next_special_moves)

    def playthrough_training_data(self,
                                  game: SelectionGame,
                                  num_reads,
                                  depth=float('inf'),
                                  policy_noise=None,
                                  ):
        """
        plays through game (or goes until depth), calling self.generate training data at each step
            uses target_policy to make moves (obtained from guided MCTS UCT search)
        Args:
            game: starting game state
            num_reads: number of reads to make in each UCT search
            depth: max depth to explore to
            policy_noise: if specified, adds noise to a distribution for exploration
                this should not be necessary, as MCTS handles the exploration
        Returns:
        """
        while (depth > 0) and (not game.is_terminal()):
            policy, _, (selection_moves, special_moves) = self.generate_training_data(game=game, num_reads=num_reads)

            if policy_noise is not None:
                policy = policy_noise(policy)
            move_idx = torch.multinomial(policy, num_samples=1).item()
            move = (selection_moves + special_moves)[move_idx]
            game = game.make_move(move)
            depth -= 1

    def policy_value_evaluator(self, game: SelectionGame, selection_moves, special_moves):
        # permuted values
        policy, values = self.network.forward(observation=game.batch_obs,
                                              selection_moves=selection_moves,
                                              special_moves=special_moves,
                                              softmax=True,
                                              )
        policy = policy.flatten()
        values = values.flatten()
        perm = game.permutation_to_standard_pos
        if perm is not None:
            values = values[perm]
        return policy.detach().numpy(), values.detach().numpy()

    def training_step(self, batch_size, shift_to_zero=True):
        sample = self.buffer.sample(batch_size)
        self.optim.zero_grad()
        policy_loss = torch.zeros(1)
        value_loss = torch.zeros(1)
        for game_rep, target_policy, target_values in sample:
            game = self.GameClass.from_representation(game_rep)
            game: SelectionGame
            policy, values = self.network.forward(observation=game.batch_obs,
                                                  selection_moves=list(game.valid_selection_moves()),
                                                  special_moves=list(game.valid_special_moves()),
                                                  softmax=True)
            policy_loss += self.policy_loss(prob_dist_pred=policy,
                                            probability_dist_targets=target_policy,
                                            shift_to_zero=shift_to_zero,
                                            )
            value_loss += self.value_loss(predicitons=values,
                                          targets=target_values,
                                          )
        policy_loss = policy_loss/batch_size
        value_loss = value_loss/batch_size

        overall_loss = value_loss + policy_loss
        overall_loss.backward()
        self.optim.step()
        return overall_loss.item(), policy_loss.item(), value_loss.item()

    def test_on_agents(self,
                       game,
                       testing_agents,
                       num_test_games,
                       trial_names=None,
                       testing_depth=float('inf'),
                       all_possible_perms=None,
                       ):
        """
        tests current policy against testing agents, returns testing dict of outcomes
        Args:
            game: starting game
            testing_agents: iterable of lists of agents to test against
                each list must have game.num_players-1 agents
            num_test_games: number of games to evaluate against each
            trial_names: names of trials to test against (if none, enumerate them)
            testing_depth: depth to send into play_game (depth to run before returning None)
            all_possible_perms: possible permutations to use to reorder test agents and self
                self is inserted into index 0
                if None, chooses a random perm
        Returns:
            testing dict of {agent name: outcome}
        """
        testing_dict = dict()
        testing_agents = list(testing_agents)
        if trial_names is None:
            trial_names = range(len(testing_agents))
        if all_possible_perms is None:
            all_possible_perms = [None for _ in range(len(testing_agents))]
        for name, agents, possible_perms in zip(trial_names, testing_agents, all_possible_perms):
            testing_dict[name] = []
            for _ in range(num_test_games):
                alg_list = [self] + list(agents)
                if possible_perms is not None:
                    perm = possible_perms[torch.randint(0, len(possible_perms), (1,)).item()]
                else:
                    perm = list(torch.randperm(len(alg_list)))

                self_idx = perm.index(0)
                alg_list = [alg_list[i] for i in perm]
                outcomes, _ = play_game(game=game,
                                        alg_list=alg_list,
                                        n=1,
                                        save_histories=False,
                                        depth=testing_depth,
                                        )
                outcome = outcomes[0]
                test = {
                    'perm': perm,
                    'outcome': outcome,
                    'self_outcome': outcome[self_idx] if outcome is not None else None,
                }
                testing_dict[name].append(test)
        return testing_dict

    def epoch(self,
              game: SelectionGame,
              batch_size,
              minibatch_size=None,
              num_reads=None,
              depth=float('inf'),
              policy_noise=None,
              save_epoch_info=True,
              testing_game=None,
              testing_agents=None,
              testing_trial_names=None,
              num_test_games=10,
              testing_depth=None,
              testing_possible_perms=None,
              ):
        """
        runs an epoch, including training data gen, training, and testing
        Args:
            game: game to start with for training data generation
            batch_size: overall batch size to use
            minibatch_size: batch size to use for each gradient
            num_reads: num_reads to send when doing MCTS
            depth: max depth to put root node at when playing training game
            policy_noise: noise to add to exploration policy (should probably stay none)
            save_epoch_info: whether to save this epoch's info
            testing_agents: list of lists of agents to train aganst
                each list must have game.num_players-1 agents to act as opponents
            testing_trial_names: names of testing trails (if None, just enumerates)
            testing_game: game to start from in testing
            num_test_games: number of trials to use in each game
            testing_depth: depth to do each testing game to
            testing_possible_perms:  possible permutations to use to reorder test agents and self
                self is inserted into index 0 for each trial
                if None, chooses a random perm each time
        Returns:
            epoch_info
        """
        epoch_info = {
            'epoch': self.info['epochs'],
        }
        if num_reads is None:
            num_reads = self.default_num_reads
        if minibatch_size is None:
            minibatch_size = batch_size
        self.playthrough_training_data(game=game,
                                       num_reads=num_reads,
                                       depth=depth,
                                       policy_noise=policy_noise,
                                       )
        overall_loss, policy_loss, value_loss = 0, 0, 0
        for i in range(0, batch_size, minibatch_size):
            # truncate last batch if larger than minibatch
            tinybatch_size = min(batch_size - i, minibatch_size)
            overall_l, policy_l, value_l = self.training_step(batch_size=minibatch_size)
            overall_loss += overall_l*tinybatch_size
            policy_loss += policy_l*tinybatch_size
            value_loss += value_l*tinybatch_size
        overall_loss = overall_loss/batch_size
        policy_loss = policy_loss/batch_size
        value_loss = value_loss/batch_size

        if testing_agents is not None:
            if testing_depth is None:
                testing_depth = depth
            if testing_game is None:
                testing_game = game
            testing_dict = self.test_on_agents(game=testing_game,
                                               num_test_games=num_test_games,
                                               testing_agents=testing_agents,
                                               trial_names=testing_trial_names,
                                               testing_depth=testing_depth,
                                               all_possible_perms=testing_possible_perms,
                                               )
            epoch_info['testing'] = testing_dict
        epoch_info['buffer_size'] = len(self.buffer)
        epoch_info['overall_loss'] = overall_loss
        epoch_info['policy_loss'] = policy_loss
        epoch_info['value_loss'] = value_loss
        self.info['epochs'] += 1
        if save_epoch_info:
            self.info['epoch_infos'].append(epoch_info)
        return epoch_info

    def get_policy_value(self,
                         game: SelectionGame,
                         selection_moves=None,
                         special_moves=None,
                         ):
        if selection_moves is None:
            selection_moves = list(game.valid_selection_moves())
        if special_moves is None:
            special_moves = list(game.valid_special_moves())
        if self.use_mcts_in_testing:
            policy, values = UCT_search(game=game,
                                        num_reads=self.default_num_reads,
                                        policy_value_evaluator=self.policy_value_evaluator,
                                        )
            return torch.tensor(policy), torch.tensor(values)
        else:
            policy, values = self.network.forward(observation=game.batch_obs,
                                                  selection_moves=selection_moves,
                                                  special_moves=special_moves,
                                                  softmax=True,
                                                  )
            policy = policy.flatten()
            values = values.flatten()
            perm = game.permutation_to_standard_pos
            if perm is not None:
                values = values[perm]
            return policy.detach(), values.detach()


if __name__ == '__main__':
    import math
    import numpy as np, random

    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    from aleph0.examples.tictactoe import Toe
    from aleph0.algs import Human
    from aleph0.networks.architect import AutoTransArchitect
    from aleph0.networks.buffers import ReplayBufferDiskStorage

    game = Toe()

    alg = AlephZero(network=AutoTransArchitect(sequence_dim=game.sequence_dim,
                                               selection_size=game.selection_size,
                                               additional_vector_dim=game.get_obs_vector_shape(),
                                               underlying_set_shapes=game.get_underlying_set_shapes(),
                                               underlying_set_sizes=game.underlying_set_sizes(),
                                               special_moves=game.special_moves,
                                               num_players=game.num_players,
                                               encoding_nums=(10, 10),
                                               base_periods_pre_exp=[-math.log(2), -math.log(2)],
                                               embedding_dim=64,
                                               dim_feedforward=128,
                                               dropout=0,
                                               num_layers=4,
                                               ),
                    replay_buffer=ReplayBufferDiskStorage(storage_dir='TEMP', capacity=1000, ),
                    GameClass=Toe,
                    )

    game = game.make_move(next(game.get_all_valid_moves()))
    game = game.make_move(next(game.get_all_valid_moves()))
    game = game.make_move(next(game.get_all_valid_moves()))
    game = game.make_move(((2, 1),))
    # there is one winning move for player 0, and the rest are losing
    # aleph should learn to only play the winning move
    print(game)
    for epoch in range(20):
        alg.generate_training_data(game=game, num_reads=alg.default_num_reads)
        alg.training_step(batch_size=16)
        print([torch.round(t, decimals=2) for t in alg.get_policy_value(game=game)])

    for epoch in range(200):
        epoch_info = alg.epoch(game=game,
                               batch_size=64,
                               minibatch_size=16,
                               )
        print(epoch, end='\r')
    # clear buffer
    alg.clear()

    print(play_game(Toe(), [Human(), alg])[0])
