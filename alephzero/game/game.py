class SubsetGame:
    """
    base class for an N-dimensional board game with K players
    each player makes a move by selecting a subset of the board positions of fixed size
        the motivation of this is a 'pick-place' game where the subset size is 2, such as chess or jenga
    this should be IMMUTABLE, all subclass methods like (make move, etc.) should return a copy of self with no shared info

    each state is associated with
        a shape (D1,...,DN, *) board
        a shape (D1,...,DN, N) 'position' board that keeps track of each index's coordinates in each of the N dimensions
        a T dimensional vector with additional game information
    T and N are fixed, while Di can change (sequential data
    all information required to play the game optimally should be encoded in this vector

    Note that the game need not be deterministic, the only requirement is that the winning strategy is only dependent
        on the current state, as opposed to past values

    for example,
    chess would be represented with
        a shape (8,8) board, where each square contains an integer encoding the identity of the piece
            the integer must also encode relevant information like an unmoved king/rook (for castling), etc
        a shape (8,8,2) position board P where entry P[i,j] simply contains the vector (i,j)
        a T dimensional vector with information such as the current player, the time since last captured piece, etc.

    jenga would be represented with
        a shape (H,3,3,E) board, H is the tower height+1, E contains information about the piece,
            such as cartesian position, rotation
        a shape (H,3,3,3) position board P where entry P[h,i,j] contains [h,i-1,j-1]
            (if we want the 'center' piece to be 0)
        a T dimensional vector with information such as the current player
    """
    COMPRESSABLE = False

    def __init__(self, current_player, subset_size, special_moves):
        """
        Args:
            current_player: player whose move it is
            subset_size: number of indices to pick to complete a normal move
            special_moves: list of special moves (i.e. END_TURN) that are not a selection of indices
                each move must have their own unique index that also cannot be a possible subset of indices
                    i.e. avoid 'special moves' like ((1,2),(2,3)), and instead name them things like 'END_TURN'
        """
        self.current_player = current_player
        self.subset_size = subset_size
        self.special_moves = special_moves

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
        raise NotImplementedError

    def possible_special_moves(self):
        """
        returns iterable of special moves possible from current position
        MUST BE DETERMINISTIC, always return moves in same order
        Returns: boolean
        """
        if not self.special_moves:
            return iter(())
        else:
            raise NotImplementedError

    def get_board_shape(self):
        """
        Returns: (D1,...,DN) for current board
        """
        raise NotImplementedError

    def get_piece_shape(self):
        """
        Returns: (*)
        """
        raise NotImplementedError

    @property
    def representation(self):
        """
        Returns: (board, position, info vector)
            of shapes (D1,...,DN, *), (D1,...,DN, N), (T,))
        should return clones of any internal variables
        """
        raise NotImplementedError

    @staticmethod
    def from_representation(representation):
        """
        returns a SubsetGame instance from the output of self.get_representation
        Args:
            representation: output of self.get_representation
        Returns: SubsetGame object
        """
        raise NotImplementedError

    def make_move(self, move):
        """
        gets resulting SubsetGame object of taking specified move from this state
        this may not be deterministic,
        cannot be called on terminal states
        Args:
            move: a subset of the possible board indices, a tuple of N-tuples
            or END_TURN token for ending turn
        Returns:
            copy of SubsetGame that represents the result of taking the move
        """
        raise NotImplementedError

    def is_terminal(self):
        """
        returns if current game has terminated
        CANNOT BE PROBABILISTIC
            if there is a probabilistic element to termination,
                the probabilities must be calculated upon creation of this object and stored
        Returns: boolean
        """
        raise NotImplementedError

    def get_result(self):
        """
        can only be called on terminal states
        returns an outcome for each player
        Returns: K-tuple of outcomes for each player
            outcomes are generally in the range [0,1] and sum to 1
            i.e. in a 1v1 game, outcomes would be (1,0) for p0 win, (0,1) for p1, and (.5,.5) for a tie
            in team games this can be changed to give teammates the same reward, and have the sum across teams be 1
        """
        raise NotImplementedError

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # NOT REQUIRED TO IMPLEMENT (either extra, or current implementation works fine)        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get_view(self, player):
        """
        gets view of game from another player
            can involve flipping board dimensions (i.e. in chess)
        must be true that self.get_view(player2).get_view(self.player) is equivalent to self
        note that this does not affect the player whose turn it is or anything
            this can be thought of as looking at the board from someone else's angle

        this is useful for games like chess, where orientation matters, and it is useful to always learn optimal moves
            when the board is oriented to face the player whose move it is

        by default, this returns self
        Args:
            player: player index (<K)
        Returns:
            a SubsetGame object clone
        """
        return self

    def clone(self):
        return self.from_representation(self.representation)

    @property
    def compressed_rep(self):
        """
        gets compressed represenation of game (if possible to save space)
        Returns: compressed represntation object
        """
        raise NotImplementedError

    @staticmethod
    def from_compressed_rep(compressed_rep):
        """
        returns a SubsetGame instance from the output of self.compressed_rep
        Args:
            representation: output of self.compressed_rep
        Returns: SubsetGame object
        """
        raise NotImplementedError

    def get_all_valid_moves(self, move_prefix=(), check_special=True):
        """
        gets all possible moves
        Args:
            move_prefix: moves selected so far,
        Returns:
            iterable of (self.subset_size tuples of N tuples)
        """
        if check_special:
            for move in self.possible_special_moves():
                yield move

        if len(move_prefix) == self.subset_size:
            yield move_prefix
        else:
            for next_move in self.get_valid_next_selections(move_prefix=move_prefix):
                new_prefix = move_prefix + (next_move,)
                for valid_move in self.get_all_valid_moves(move_prefix=new_prefix, check_special=False):
                    yield valid_move

    def symmetries(self, policy_vector):
        """
        Args:
            policy_vector: torch vector of size get_all_valid_moves
        Returns:
            iterable of (SubsetGame, policy_vector) objects that encode equivalent games
            Note that the policy_vector input will likely need to be permuted to match each additional symmetry
            i.e. rotated boards in tictactoe
            used for training algorithm
        """
        yield (self, policy_vector)

    def get_choices_on_dimension(self, dim):
        """
        gets all valid moves on specified dimension
        this is for if on this dimension, valid moves are independent of the previous move choices
            i.e. if 'place' was always a choice of 3 moves like jenga
        this does not need to be implemented, as it is not always true
            (i.e. chess place moves are dependent on which piece was selected)
        Args:
            dim: dimension to inspect
        Returns:
            iterable of N tuples indicating which choices are valid on this dimension
        """
        raise NotImplementedError


class PickGame(SubsetGame):
    """
    cmmmon board game setup
    select a square at each turn (tic tac toe style)
    """

    def __init__(self, current_player,special_moves):
        super().__init__(current_player=current_player,subset_size=1, special_moves=special_moves)


class PickPlaceGame(SubsetGame):
    """
    cmmmon board game setup
    select a piece to move, and a place to move it
    """

    def __init__(self,current_player, special_moves):
        super().__init__(current_player=current_player,subset_size=2, special_moves=special_moves)
