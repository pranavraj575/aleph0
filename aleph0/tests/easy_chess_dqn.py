if __name__ == '__main__':
    import torch
    from aleph0.examples.chess import Chess2d
    from aleph0.algs import Human, DQNAlg_from_game, play_game
    from aleph0.examples.chess.game import Board, P
    torch.random.manual_seed(0)

    test_game = Chess2d()
    alg = DQNAlg_from_game(game=test_game)
    board = torch.ones(Board.BOARD_SHAPE, dtype=torch.long)*P.EMPTY
    board[-1, -1] = P.as_player(piece=P.KING, player=P.P1)
    board[0, 0] = P.ROOK
    board[0, 1] = P.ROOK
    board[0,2] = P.KING
    game = Chess2d(initial_board=Board(board=board))
    for i in range(100):
        print(i,end='\t\t\r')
        alg.train_episode(game=game,
                          epsilon=.01,
                          depth=50,
                          )
    quit()
    outcome = play_game(game, [alg, Human()])
    print(outcome)
