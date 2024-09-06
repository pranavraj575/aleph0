def add_experiment_args(parse,
                        default_epochs=2000,
                        default_ckpt_freq=25,
                        default_smooth_rad=10,
                        ):
    parse.add_argument('--ident', action='store', required=False, default='ult_toe_test',
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


def add_trans_args(parse,
                   default_embed_dim=128,
                   default_num_layers=4,
                   default_dim_feedforward=128,
                   default_dropout=.1,
                   ):
    parse.add_argument('--embed-dim', type=int, required=False, default=default_embed_dim,
                       help="transformer embed dim")
    parse.add_argument('--num-layers', type=int, required=False, default=default_num_layers,
                       help="transformer num layers")
    parse.add_argument('--dim-feedforward', type=int, required=False, default=default_dim_feedforward,
                       help="transformer feedforward dimensiton")
    parse.add_argument('--dropout', type=float, required=False, default=default_dropout,
                       help="transformer dropout")


def get_trans_ident(args):
    ident = ''
    ident += '_embed_dim_' + str(args.embed_dim)
    ident += '_lyrs_' + str(args.num_layers)
    ident += '_drop_' + str(float(args.dropout))
    ident += '_dim_ff_' + str(args.dim_feedforward)
    return ident
