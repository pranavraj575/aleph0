import torch

from aleph0.game import FixedSizeSelectionGame


class Toe(FixedSizeSelectionGame):
    EMPTY = 2
    P0 = 0
    P1 = 1

    def __init__(self, current_player=P0, board=None):
        super().__init__(
            num_players=2,
            current_player=current_player,
            subset_size=1,
            special_moves=[],
        )
        if board is None:
            board = self.EMPTY*torch.ones((3, 3), dtype=torch.long)
        self.board = board

    def get_valid_next_selections(self, move_prefix=()):
        for (i, j) in zip(*torch.where(torch.eq(self.board, self.EMPTY))):
            yield i.item(), j.item()

    @staticmethod
    def fixed_obs_shape():
        return ((3, 3),), (3, 3, 2), (1,)

    @staticmethod
    def underlying_set_sizes():
        return (3,)

    def possible_move_cnt(self):
        print('here',super().possible_move_cnt())
        return 9

    def index_to_move(self, idx):
        return ((idx//3, idx%3),)

    def move_to_idx(self, move):
        return move[0][0]*3 + move[0][1]

    @staticmethod
    def invert_player(player):
        return 1 - player

    @property
    def permutation_to_standard_pos(self):
        if self.current_player == self.P1:
            return [1, 0]
        else:
            return [0, 1]

    @property
    def representation(self):
        I = torch.cat((torch.arange(3).view((3, 1, 1)), torch.zeros((3, 1, 1))), dim=-1)
        J = torch.cat((torch.zeros((1, 3, 1)), torch.arange(3).view((1, 3, 1))), dim=-1)
        return self.board.clone(), (I + J), torch.tensor([self.current_player])

    @property
    def observation(self):
        if self.current_player == self.P0:
            B, P, T = self.representation
        else:
            B, P, T = self.representation
            p0s = torch.where(torch.eq(B, self.P0))
            p1s = torch.where(torch.eq(B, self.P1))
            B[p0s] = self.P1
            B[p1s] = self.P0
        return (B,), P, T

    @staticmethod
    def from_representation(representation):
        board, _, vec = representation
        return Toe(board=board, current_player=vec.item())

    def make_move(self, local_move):
        board = self.board.clone()
        board[local_move[0]] = self.current_player
        return Toe(current_player=self.invert_player(self.current_player), board=board)

    def is_terminal(self):
        # board is terminal if either there are no moves left, or there
        return (self.EMPTY not in self.board) or self.get_result() != (.5, .5)

    def get_result(self):
        for test, ret in ((self.P0, (1., 0.)),
                          (self.P1, (0., 1.))):
            for k in range(3):
                if (torch.all(torch.eq(self.board[k, :], test)) or
                        torch.all(torch.eq(self.board[:, k], test))
                ):
                    return ret
            if (torch.all(torch.eq(self.board[range(3), range(3)], test)) or
                    torch.all(torch.eq(self.board[range(3), [-1 - i for i in range(3)]], test))):
                return ret
        return (.5, .5)

    def __str__(self):
        return '---\n' + str(self.board.numpy()
                             ).replace(' ', ''
                                       ).replace('[', ''
                                                 ).replace(']', ''
                                                           ).replace('-1', ' '
                                                                     ).replace('0', 'X'
                                                                               ).replace('1', 'O'
                                                                                         ).replace('2', '.'
                                                                                                   ) + '\n---'


if __name__ == '__main__':
    toe = Toe()
    while True:
        toe = toe.make_move(local_move=next(toe.get_all_valid_moves()))
        print(toe)

        if toe.is_terminal(): break
    print(toe.get_result())
    toe.possible_move_cnt()
