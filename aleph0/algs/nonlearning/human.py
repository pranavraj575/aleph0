import torch

from aleph0.game import SelectionGame
from aleph0.algs.algorithm import Algorithm


class Human(Algorithm):
    """
    takes user input to make moves
    """

    def get_policy_value(self, game: SelectionGame, moves=None):
        if moves is None:
            moves = list(game.get_all_valid_moves())
        move_prefix = ()
        selected = None
        while True:
            game.render()
            print('current player', game.current_player)
            next_choices = [choice
                            for choice in game.get_valid_next_selections(move_prefix=move_prefix)
                            if move_prefix + (choice,) in [mv[:len(move_prefix) + 1] for mv in moves]
                            ]
            next_specials = list(game.valid_special_moves())
            if move_prefix:
                all_choices = next_choices
            else:
                all_choices = next_specials + next_choices
            print('possible choices:')
            for i, choice in enumerate(all_choices):
                print(str(i) + ':', choice)
            if move_prefix:
                print('-1: backspace')
            selection = input('choose index: ')
            if selection.isnumeric():
                if not (len(selection) > 1 and selection.startswith('0')):
                    selection = int(selection)
                    if selection < len(all_choices):
                        move_choice = all_choices[selection]
                        if move_choice in next_specials:
                            selected = move_choice
                        else:
                            move_prefix += (move_choice,)
                            if move_prefix in moves:
                                selected = move_prefix
            else:
                # backspace
                if len(move_prefix) > 0:
                    move_prefix = move_prefix[:-1]
            if selected is not None:
                disp_game = game.make_move(selected)
                disp_game.current_player=game.current_player
                disp_game.render()
                print('next state:')
                print('move made:', selected)
                if input('redo? [y/n]: ').lower() == 'y':
                    move_prefix = ()
                    selected = None
                else:
                    break
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
