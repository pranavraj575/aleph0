import torch

from alephzero.game.game import PickGame


class Toe(PickGame):
    EMPTY = -1
    P0 = 0
    P1 = 1

    def __init__(self, current_player=P0, board=None):
        super().__init__(current_player=current_player, special_moves=[])
        if board is None:
            board = self.EMPTY*torch.ones((3, 3))
        self.board = board

    def get_valid_next_selections(self, move_prefix=()):
        for (i, j) in zip(*torch.where(torch.eq(self.board, self.EMPTY))):
            yield i.item(), j.item()

    def get_board_shape(self):
        return (3, 3)

    def get_piece_shape(self):
        return ()

    @staticmethod
    def invert_player(player):
        return 1 - player

    @property
    def representation(self):
        I = torch.cat((torch.arange(3).view((3, 1, 1)), torch.zeros((3, 1, 1))), dim=-1)
        J = torch.cat((torch.zeros((1, 3, 1)), torch.arange(3).view((1, 3, 1))), dim=-1)
        return self.board.clone(), (I + J), torch.tensor([self.player])

    @staticmethod
    def from_representation(representation):
        board, _, vec = representation
        return Toe(board=board, current_player=vec.item())

    def make_move(self, move):
        board = self.board.clone()
        board[move[0]] = self.current_player
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


if __name__ == '__main__':
    toe = Toe()
    while True:
        toe = toe.make_move(move=next(toe.get_all_valid_moves()))
        print(toe.board)
        if toe.is_terminal(): break
    print(toe.get_result())
