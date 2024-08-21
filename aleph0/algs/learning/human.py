import torch

from aleph0.game import SelectionGame
from aleph0.algs.algorithm import Algorithm


class Human(Algorithm):
    """
    takes user input to make moves
    """

    def get_policy_value(self, game: SelectionGame, selection_moves=None, special_moves=None):
        if selection_moves is None:
            selection_moves = list(game.get_all_valid_moves())
        if special_moves is None:
            special_moves = list(game.valid_special_moves())
        move_prefix = ()
        selected = None

        # idxs is (D1,...,DN,N) array of indexes
        _, idxs, _ = game.observation
        while True:
            game.render()
            print('current player', game.current_player)
            temp = [select_move[len(move_prefix)]
                    for select_move in selection_moves
                    if move_prefix == select_move[:len(move_prefix)]
                    ]
            selection_next_choices = []
            for item in temp:
                if item not in selection_next_choices:
                    selection_next_choices.append(item)
            if move_prefix:
                all_choices = selection_next_choices
            else:
                all_choices = selection_next_choices + special_moves
            print('possible choices:')
            for i, choice in enumerate(selection_next_choices):
                # we want to display the index of the choice
                print(str(i) + ':', idxs[choice].numpy())
            if not move_prefix:
                for i, choice in enumerate(special_moves):
                    print(str(i + len(selection_next_choices)) + ':', choice)
            if move_prefix:
                print('-1: backspace')
            selection = input('choose index: ')
            if selection.isnumeric():
                if not (len(selection) > 1 and selection.startswith('0')):
                    selection = int(selection)
                    if selection < len(all_choices):
                        move_choice = all_choices[selection]
                        if move_choice in special_moves:
                            selected = move_choice
                        else:
                            move_prefix += (move_choice,)
                            if move_prefix in selection_moves:
                                selected = move_prefix
            else:
                # backspace
                if len(move_prefix) > 0:
                    move_prefix = move_prefix[:-1]
            if selected is not None:
                disp_game = game.make_move(selected)
                disp_game.current_player = game.current_player
                disp_game.render()
                print('next state:')
                if move_prefix:
                    print('move made: (', end='')
                    for s in selected[:-1]:
                        print(idxs[s].numpy(), end=', ')
                    print(idxs[selected[-1]].numpy(), end='')
                    print(')')
                else:
                    print('move made:', selected)
                if input('redo? [y/n]: ').lower() == 'y':
                    move_prefix = ()
                    selected = None
                else:
                    break
        moves = selection_moves + special_moves
        dist = torch.zeros(len(moves))
        dist[moves.index(selected)] = 1
        return dist, None


if __name__ == '__main__':
    from aleph0.examples.tictactoe import Toe

    # if run on initial game, takes a while, then returns that every move is a tying move
    # distribution is uniform over all moves, and value is (.5,.5)
    game = Toe()
    me = Human()
    while not game.is_terminal():
        dist, _ = me.get_policy_value(game)
        move_idx = torch.multinomial(dist, 1)
        game = game.make_move(list(game.get_all_valid_moves())[move_idx])
    print('game over, result:')
    print(game.get_result())
