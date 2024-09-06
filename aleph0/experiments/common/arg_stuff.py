def add_experiment_args(parse,
                        ident,
                        default_epochs=2000,
                        default_ckpt_freq=25,
                        default_smooth_rad=10,
                        default_test_games=5,
                        default_test_freq=1,
                        ):
    parse.add_argument('--ident', action='store', required=False, default=ident,
                       help="folder name")
    parse.add_argument('--epochs', type=int, required=False, default=default_epochs,
                       help="number of epochs")

    parse.add_argument('--ckpt-freq', type=int, required=False, default=default_ckpt_freq,
                       help="frequency to checkpoint")

    parse.add_argument('--plot', action='store_true', required=False,
                       help="whether to plot")
    parse.add_argument('--smooth-radius', type=int, required=False, default=default_smooth_rad,
                       help="radius of smoothing window of plot")
    parse.add_argument('--play', action='store_true', required=False,
                       help="whether to play game with alg")
    parse.add_argument('--reset', action='store_true', required=False,
                       help="reset training")

    parse.add_argument('--num-test-games', type=int, required=False, default=default_test_games,
                       help="number of test games to use")
    parse.add_argument('--test-freq', type=int, required=False, default=default_test_freq,
                       help="frequency to do test games")


def add_trans_args(parse,
                   default_embed_dim=128,
                   default_num_heads=4,
                   default_num_layers=4,
                   default_dim_feedforward=128,
                   default_dropout=.1,
                   ):
    parse.add_argument('--embed-dim', type=int, required=False, default=default_embed_dim,
                       help="transformer embed dim")
    parse.add_argument('--num-heads', type=int, required=False, default=default_num_heads,
                       help="transformer num layers")

    parse.add_argument('--num-layers', type=int, required=False, default=default_num_layers,
                       help="transformer num layers")
    parse.add_argument('--dim-feedforward', type=int, required=False, default=default_dim_feedforward,
                       help="transformer feedforward dimensiton")
    parse.add_argument('--dropout', type=float, required=False, default=default_dropout,
                       help="transformer dropout")


def get_trans_ident(args):
    ident = ''
    ident += '_embed_dim_' + str(args.embed_dim)
    ident += '_heads_' + str(args.num_heads)
    ident += '_lyrs_' + str(args.num_layers)
    ident += '_drop_' + str(float(args.dropout))
    ident += '_dim_ff_' + str(args.dim_feedforward)

    return ident


def add_aleph_args(parse,
                   default_num_reads=420,
                   default_buffer_capacity=10000,
                   default_batch_size=512,
                   default_minibatch_size=256,
                   ):
    parse.add_argument('--num-reads', type=int, required=False, default=default_num_reads,
                       help="num reads for mcts in alephzero")
    parse.add_argument('--buffer-capacity', type=int, required=False, default=default_buffer_capacity,
                       help="buffer capacity")
    parse.add_argument('--batch-size', type=int, required=False, default=default_batch_size,
                       help="batch size alephzero")
    parse.add_argument('--minibatch-size', type=int, required=False, default=default_minibatch_size,
                       help="minibatch size for alephzero")


def get_aleph_ident(args):
    ident = ''
    ident += '_mctsreads_' + str(args.num_reads)
    ident += '_cap_' + str(args.buffer_capacity)
    ident += '_btch_' + str(args.batch_size)
    ident += '_mini_' + str(args.minibatch_size)
    return ident
