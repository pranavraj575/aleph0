import torch

from aleph0.game.selection_game import SelectionGame


def play_game(game: SelectionGame, alg_list, n=1, save_histories=True):
    """
    plays n games starting from game state, using each alg in alg_list as player
    repeatedly samples moves from alg.get_policy_value
    Args:
        game: SubsetGame with K players, should prob be non-terminal
        alg_list: list of K algorithms to play with
        n: number of games to play
        save_histories: whether to return histories
            each history is a list of (game, action, next_game)
    Returns:
        list of outcomes, list of histories if save_histories (otherwise None)
    """
    outcomes = []
    histories = []
    for _ in range(n):
        temp = game.clone()
        history = []
        while not temp.is_terminal():
            player = temp.current_player
            alg = alg_list[player]
            selection_moves = list(temp.valid_selection_moves())
            special_moves = list(temp.valid_special_moves())
            valid_moves = selection_moves + special_moves
            dist, _ = alg.get_policy_value(game=temp,
                                           selection_moves=selection_moves,
                                           special_moves=special_moves,
                                           )

            move_idx = torch.multinomial(dist, 1).item()
            next_temp = temp.make_move(valid_moves[move_idx])
            if save_histories:
                history.append((game, valid_moves[move_idx], next_temp))
            temp = next_temp
        outcomes.append(temp.get_result())
        if save_histories:
            histories.append(history)
    if save_histories:
        return outcomes, histories
    else:
        return outcomes, None


if __name__ == '__main__':
    from aleph0.examples.tictactoe import Toe
    from aleph0.algs import Exhasutive, Human, Randy

    game = Toe()
    print('outcome of a random game:')
    print(play_game(game, [Randy(), Randy()])[0])
    print('play against a random agent:')
    print(play_game(game, [Human(), Randy()])[0])
    print('you cannot win this')
    print(play_game(game, [Human(), Exhasutive()])[0])

    print('game of bad vs perfect')
    print(play_game(game, [Randy(), Exhasutive()])[0])
    print('perfect game')
    print(play_game(game, [Exhasutive(), Exhasutive()])[0])
