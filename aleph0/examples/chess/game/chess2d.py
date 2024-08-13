import torch, copy

from aleph0.examples.chess.game.piece import P
from aleph0.examples.chess.game.board import Board
from aleph0.examples.chess.game.timeline import Timeline
from aleph0.examples.chess.game.multiverse import Multiverse
from aleph0.examples.chess.game.chess5d import Chess5d

from aleph0.game import FixedSizeSelectionGame


class Chess2d(Chess5d, FixedSizeSelectionGame):
    def __init__(self,
                 initial_board=None,
                 initial_timeline=None,
                 current_player=P.P0,
                 first_player=P.P0,
                 check_validity=False,
                 save_moves=True,
                 term_ev=None,
                 ):
        super().__init__(initial_board=initial_board,
                         initial_timeline=initial_timeline,
                         current_player=current_player,
                         first_player=first_player,
                         check_validity=check_validity,
                         save_moves=save_moves,
                         term_ev=term_ev,
                         )

    def _piece_possible_moves(self, global_idx, castling=True):
        # only return moves that do not jump time-dimensions
        for end_idx in super()._piece_possible_moves(global_idx, castling=castling):
            if end_idx[:2] == global_idx[:2]:
                yield end_idx

    def material_draw(self):
        board = self.get_current_board()
        if self.present_player_in_check():
            # then it is not a draw
            return False
        for idx in board.all_pieces():
            piece = board.get_piece(idx)
            if P.piece_id(piece=piece) != P.KING:
                return False
        return True

    def get_current_board(self):
        return self.multiverse.get_board((self.multiverse.max_length - 1, 0))

    def get_current_timeline(self):
        return self.multiverse.get_timeline(0)

    def wrap_move(self, move, td_idx=None):
        if td_idx is None:
            td_idx = (self.multiverse.max_length - 1, 0)
        idx, end_idx = move
        return td_idx + idx, td_idx + end_idx

    def convert_to_local_idx(self, global_idx):
        if self.current_player == P.P0:
            return global_idx
        else:
            (i, j) = global_idx
            I, J = Board.BOARD_SHAPE
            return (I - i - 1, J - j - 1)

    def convert_to_global_idx(self, local_idx):
        # these functions are the same in this case, where we only care about range 0 to 8
        return self.convert_to_local_idx(global_idx=local_idx)

    def flipped(self):
        out = Chess2d(
            initial_timeline=self.get_current_timeline().flipped(),
            current_player=P.flip_player(self.current_player),
            first_player=P.flip_player(self.first_player),
            check_validity=self.check_validity,
            save_moves=self.save_moves,
            term_ev=self.term_ev,
        )
        out.turn_history = [[(Chess5d._flip_move(move), [-dim for dim in dims_spawned])
                             for (move, dims_spawned) in turn]
                            for turn in self.turn_history]
        return out

    def prune_timeline(self):
        range = self.get_current_timeline().get_time_range()
        if range[1] - range[0] > 3:
            board_list = self.get_current_timeline().board_list
            start_idx = range[1] - 3
            self.multiverse = Multiverse(
                main_timeline=Timeline(
                    board_list=board_list[-3:],
                    start_idx=start_idx,
                )
            )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CLASS METHODS                                                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def clone(self):
        game = Chess2d(initial_timeline=self.get_current_timeline().clone(),
                       check_validity=self.check_validity,
                       save_moves=self.save_moves,
                       current_player=self.current_player,
                       first_player=self.first_player,
                       term_ev=self.term_ev,
                       )
        game.turn_history = copy.deepcopy(self.turn_history)
        game._prune_history()
        return game

    def render(self):
        if self.current_player == P.P1:
            game = self.flipped()
        else:
            game = self
        print(game.get_current_board().__str__())

    def get_valid_next_selections(self, move_prefix=()):
        """
        gets valid choices for next index to select
            MUST BE DETERMINISTIC
            moves must always be returned in the same order
        Args:
            move_prefix: indices selected so far, must be less than self.subsetsize
        Returns:
            iterable of N tuples indicating which additions are valid
        """
        if move_prefix == ():
            board = self.get_current_board()
            for (i, j) in board.pieces_of(self.current_player):
                yield self.convert_to_local_idx((i, j))
        else:
            local_idx, = move_prefix
            global_idx = (self.multiverse.max_length - 1, 0) + self.convert_to_local_idx(local_idx)
            for end_idx in self._piece_possible_moves(global_idx=global_idx, castling=True):
                yield self.convert_to_local_idx(end_idx[2:])

    def valid_special_moves(self):
        """
        returns iterable of special moves possible from current position
        MUST BE DETERMINISTIC, always return moves in same order
        Returns: boolean
        """
        return iter(())

    def make_move(self, local_move):
        out = self.clone()
        global_move = self.convert_to_global_move(local_move)
        global_move = self.wrap_move(global_move)
        capture, terminal = out._mutate_make_move(global_move)

        if terminal:
            out.term_ev = out._terminal_eval(mutation=False)
        else:
            # only do this if non terminal
            out._mutate_make_move(Chess2d.END_TURN)
        out.prune_timeline()
        return out

    @staticmethod
    def fixed_obs_shape():
        """
        observation is shapes ((D1,D2),), (D1,D2,2),0)
        this method returns those shapes
        """
        return (Board.BOARD_SHAPE,), (Board.BOARD_SHAPE + (2,)), (0,)

    @property
    def observation_shape(self):
        """
        observation is shapes ((D1,D2),), (D1,D2,2), 0)
        this method returns those shapes
        """
        return self.fixed_obs_shape()

    @property
    def observation(self):
        """
        observation is shapes ((D1,D2),), (D1,D2,2), 0)
        Note that D1 and D2 are fixed

        the board is
            the pieces (including empty squares)
        indexes are normal indices, from the point of view of current player
        Returns:

        """
        if self.current_player == P.P1:
            game = self.flipped()
        else:
            game = self.clone()
        board = game.get_current_board()
        board_obs = board.get_board_as_indices()
        board_shapes, index_shape, vec_shape = game.observation_shape
        (xlen, ylen) = index_shape[:2]

        # create index set
        X = torch.cat((
            torch.arange(xlen).view((xlen, 1, 1)),
            torch.zeros((xlen, 1, 1)),
        ), dim=-1)
        Y = torch.cat((
            torch.zeros((1, ylen, 1)),
            torch.arange(ylen).view((1, ylen, 1)),
        ), dim=-1)
        return (board_obs,), X + Y, torch.zeros(vec_shape)

    @staticmethod
    def num_observation_boards():
        """
        Returns: number of boards in (D1,...,DN, *1),(D1,...,DN, *2),...)
        """
        return 1

    @staticmethod
    def underlying_set_sizes():
        """
        returns number of possible distinct elements of each underlying set, if finite
        """
        return (P.TOTAL_PIECES,)

    def possible_move_cnt(self):
        """
        all possible choices of two squars, plus an extra unused move
        """
        return Board.BOARD_SQUARES**2 + 1

    def index_to_move(self, idx):
        """
        convert idx into a valid move
        """
        # this should never happen, but just in case
        if idx == self.possible_move_cnt() - 1:
            return Chess2d.END_TURN

        I, J = Board.BOARD_SHAPE
        num_squares = Board.BOARD_SQUARES
        start_idx = idx//num_squares
        start_idx = start_idx//J, start_idx%J
        end_idx = idx%num_squares
        end_idx = end_idx//J, end_idx%J
        return (start_idx, end_idx)

    def move_to_idx(self, move):
        I, J = Board.BOARD_SHAPE
        if move == Chess2d.END_TURN:
            return self.possible_move_cnt() - 1
        (i1, j1), (i2, j2) = move
        return Board.BOARD_SQUARES*(i1*J + j1) + i2*J + j2


if __name__ == '__main__':
    from aleph0.algs import Human, play_game

    chess = Chess2d()
    print(chess.possible_move_cnt())
    print(play_game(chess, [Human(), Human()], save_histories=False))
