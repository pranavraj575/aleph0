if __name__ == '__main__':
    from aleph0.examples import Toe, UltimateToe, Chess5d, Chess2d, Jenga, JengaOne
    from aleph0.algs import play_game, Human
    import argparse

    PARSER = argparse.ArgumentParser()
    jengagroup = PARSER.add_argument_group('jenga', 'arguments for jenga')
    jengagroup.add_argument('--num-players', type=int, default=2,
                            help='number of players playing')
    game_map = {'toe': (lambda: Toe()),
                'ultimatetoe': (lambda: UltimateToe()),
                'chess5d': (lambda: Chess5d()),
                'chess2d': (lambda: Chess2d()),
                'jenga': (lambda: Jenga(num_players=args.num_players, )),
                'jengaone': (lambda: JengaOne(num_players=args.num_players, ))
                }
    gmk = list(game_map.keys())
    description_map = {'toe': 'tic tac toe',
                       'ultimatetoe': 'ultimate tic tac toe',
                       'chess5d': '5d chess with multiverse time travel',
                       'chess2d': '2d chess without multiverse time travel',
                       'jenga': 'jenga',
                       'jengaone': 'jenga with pick/place as different moves'
                       }
    PARSER.add_argument('--game', action='store', required=False, default=None,
                        choices=gmk,
                        help="game to play")
    PARSER.add_argument('--info-on-game', action='store', required=False, default=None,
                        choices=gmk,
                        help='display info for specified game and quit')
    args = PARSER.parse_args()
    if args.info_on_game is not None:
        print(description_map[args.info_on_game])
        quit()
    if args.game is None:
        print('need to specify --game {' + ','.join(gmk) + '}')
        quit()
    game=game_map[args.game]()
    play_game(game=game, alg_list=[Human() for _ in range(game.num_players)])
