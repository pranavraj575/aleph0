"""
contains Tower class
encodes a tower, has helpful methods to get state, as well as make moves
"""
from scipy.spatial import ConvexHull
import numpy as np
import torch

from aleph0.game import SelectionGame

TOLERANCE = 1e-10


def point_in_hull(point, hull, tolerance=TOLERANCE):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def which_points_in_hull(points, hull, tolerance=TOLERANCE):
    """
    returns if points np array dim (N,2) are in hull

    return N vector of booleans
    """
    # adds one to each row to dot product easier
    aug_points = np.concatenate((points, np.ones((len(points), 1))), axis=1)

    # each column of this is the distance of each projection to all the equations
    # i.e. all_dists[i,j] is the distance of the jth point on the ith equation
    all_dists = np.dot(hull.equations, aug_points.T)

    # maximum distance of each point to the equation line (if postive, that means its outside the line)
    dists = np.max(all_dists, axis=0)

    # returns which of them have all negative distances (i.e. within all the bounds)
    return dists <= tolerance


def hull_score_from_vtxs(point, vtxs):
    return hull_score(point, ConvexHull(vtxs))


def hull_score(point, hull, tolerance=TOLERANCE):
    """
    returns the 'score' of the point in the hull
    if the point lies outside of the hull, returns positive number
        score is distance to closest point in hull
        represents how far it is from being stable
    if the point lies inside, returns negative number
        score is -(the distance to the closest bound)
        represents how close it is to being out of bounds
        note that we do not need to check vertices, as the closest point will be to a line
    """

    # hull equations are set up that dot(point, eq[:-1])+eq[-1] is the signed distance from the line
    # if we augment the point with a 1, same as dot(point,eq) for each equation
    # then we can dot the entire matrix for speed
    dists = np.dot(hull.equations, np.concatenate((point, [1])))
    stable = np.all(dists <= 0)

    if stable:
        return np.max(dists)
    else:
        # in this case, the closest point will either be a vertex or a projection onto a line

        # projections to each line is the point minus (error * line vector)
        projections = point - hull.equations[:, :-1]*dists.reshape(-1, 1)

        # filter projections by the ones actually within the hull
        valid_projections = projections[which_points_in_hull(projections, hull, tolerance=tolerance)]

        # points to check are valid projections and the vertices of original hull
        points = np.concatenate((valid_projections, hull.points))

        # distance to all points
        point_dists = np.linalg.norm(points - point, axis=1)

        # return the minimum of these
        return np.min(point_dists)


class Block:
    STD_BLOCK_DIM = np.array((.075, .025, .015))  # in meters
    STD_BLOCK_SPACING = .005  # in meters

    STD_WOOD_DENSITY = 0.5  # kg/m^3

    def __init__(self,
                 pos,
                 yaw=0.,
                 block_dim=None,
                 density=None,
                 ):
        """
        represents one block
        :param pos: position, np array
        :param yaw: represents orientation

        Note: generally, roll and pitch are 0 and yaw is the only relevant one
        """
        self.pos = pos
        self.yaw = yaw
        if block_dim is None:
            block_dim = Block.STD_BLOCK_DIM
        if density is None:
            density = Block.STD_WOOD_DENSITY
        self.block_dim = block_dim
        self.density = density
        self.mass = np.prod(self.block_dim)*self.density

    def representation(self):
        """
        returns block encoded as a vector

        x,y,z,angle
        """
        return np.concatenate((self.pos, self.block_dim, [self.yaw, self.density]))

    @staticmethod
    def from_representation(representation):
        pos = representation[:3]
        block_dim = representation[3:6]
        yaw, density = representation[6:]
        return Block(pos=pos,
                     yaw=yaw,
                     block_dim=block_dim,
                     density=density,
                     )

    def vertices(self):
        """
        returns vertices of block
        :return: 8x3 array
        """

        dx, dy, dz = self.block_dim

        return np.array([[
            (X,
             Y,
             self.pos[2] + dz*(z_i - .5),
             )
            for (X, Y) in self.vertices_xy()] for z_i in range(2)]).reshape((8, 3))

    def vertices_xy(self):
        """
        returns xy projected vertices of block
        ordered in a cycle
        :return: 4x2 array
        """
        dx, dy, _ = self.block_dim
        return self.pos[:2] + np.array([[
            (dx*(x_i - .5)*np.cos(self.yaw) - dy*(y_i - .5)*np.sin(self.yaw),
             dx*(x_i - .5)*np.sin(self.yaw) + dy*(y_i - .5)*np.cos(self.yaw),
             )
            for x_i in (range(2) if y_i == 0 else range(1, -1, -1))]  # go in reverse for y_i=1
            for y_i in range(2)]).reshape((4, 2))

    def com(self):
        """
        :return: np array, center of mass (is just pos)
        """
        return self.pos

    def __eq__(self, other):
        return (np.array_equal(self.pos, other.pos) and
                self.yaw == other.yaw and
                np.array_equal(self.block_dim, other.block_dim) and
                self.density == other.density)


def random_block(L, i, pos_std=0., angle_std=0.):
    """
    creates a randomly placed block at specified level and index
    :param L: level of block in tower
    :param i: index of block in level (increases +x for even levels, +y for odd levels)
        0 is the furthest negative, 2 is the furthest positive
    :param pos_std: std randomness of position
    :param angle_std: std randomness of angle
    :return: block object
    """

    rot = (L%2)*np.pi/2  # rotate yaw if odd level

    pos = np.zeros(3)
    pos += (0, 0, (L + 0.5)*Block.STD_BLOCK_DIM[2])  # height of level + half the height of a block (since COM)

    width = Block.STD_BLOCK_DIM[1]
    if L%2:
        pos += ((i - 1)*(width + Block.STD_BLOCK_SPACING), 0, 0)
    else:
        pos += (0, (i - 1)*(width + Block.STD_BLOCK_SPACING), 0)

    rot = rot + np.random.normal(0, angle_std)
    pos = pos + (np.random.normal(0, pos_std), np.random.normal(0, pos_std), 0.)

    return Block(pos=pos, yaw=rot)


class Jenga(SelectionGame):
    """
    jenga tower representation
    list of layers, each layer is a list of three booleans
    Note: this class does not change after initalization
        all methods like "remove_block" create a new instance of the class
    """

    TOWER_DIAMETER = 2*Block.STD_BLOCK_SPACING + 3*Block.STD_BLOCK_DIM[1]  # theoretical size of a layer

    INITIAL_SIZE = 5  # number of levels in initial tower

    def __init__(self,
                 num_players,
                 current_player=0,
                 block_info=None,
                 pos_std=.001,
                 angle_std=.003,
                 ):
        """
        :param block_info: list of block triples, represents each layer
        :param default_ht: height to create tower if block info is None
        :param pos_std: stdev to use for positions if randomly initializing/randomly placing
        :param angle_std: stdev to use for angles if randomly initializing/randomly placing
        """
        super().__init__(num_players=num_players,
                         current_player=current_player,
                         subset_size=2,
                         special_moves=[],
                         )

        self.pos_std = pos_std
        self.angle_std = angle_std
        if block_info is None:
            block_info = [
                [random_block(L=level, i=i, pos_std=pos_std, angle_std=angle_std) for i in range(3)]
                for level in range(Jenga.INITIAL_SIZE)
            ]
        self.block_info = block_info

        self.calculated = False  # whether we calculated things like COM
        # compute all of these
        self.update_info()

    def update_info(self):
        """
        computation of features to avoid having to compute them each time
        """
        self.calculated = True
        # since each block has equal mass, the COM of tower is just the straight average of COMs of blocks
        # we compute COMs at each layer, the COM of 'subtowers' starting from a layer are of interest
        self.Ns = []  # number of blocks above each layer, including that layer
        self.COMs = []  # COM above each layer (inclusive, so COMs[0] is the COM of the tower, and COMs[-1] is the COM of last layer)
        self.hulls = []  # store the convex hull of each layer.

        self.block_hulls = None  # these might not be calculated

        # going backwards so we can accumulate
        N = 0  # running count
        MOMENT = np.zeros(3)  # running moment (sum of positions of blocks, not averaged yet)
        MASS = 0
        for layer in self.block_info[::-1]:
            N += sum([t is not None for t in layer])
            MOMENT += np.sum([b.com()*b.mass for b in layer if b is not None], axis=0)
            MASS += sum([b.mass for b in layer if b is not None])
            self.Ns.append(N)
            self.COMs.append(MOMENT/MASS)

            V = np.concatenate([t.vertices_xy() for t in layer if t is not None], axis=0)
            self.hulls.append(ConvexHull(V))

        self.Ns.reverse()
        self.COMs.reverse()
        self.hulls.reverse()

    def update_special_info(self):
        """
        these probably wont be necessary for EVERY tower, only run if needed
        """
        self.block_hulls = []
        for layer in self.block_info:
            hull_layer = [ConvexHull(t.vertices_xy()) for t in layer if t is not None]
            self.block_hulls.append(hull_layer)

    # ACCESS METHODS
    def get_block_hulls(self):
        if self.block_hulls is None:
            self.update_special_info()
        return self.block_hulls

    def boolean_blocks(self):
        return [[t is not None for t in layer] for layer in self.block_info]

    def num_blocks(self):
        """
        :return: number of blocks
        """
        return self.Ns[0]

    def blocks_on_level(self, L):
        """
        returns blocks on level L
        :param L: level
        :return: int
        """
        if (L == self.height() - 1) or (L == -1):
            return self.Ns[-1]
        return self.Ns[L] - self.Ns[L + 1]

    def free_valid_moves(self):
        """
        returns total 'free moves' that are valid to take
            a free move is the max number of moves left before a level is probably fallen
            a full layer has 2 free moves
            a layer with one block has 0 free moves
            a layer with 2 blocks has either 0 or 1 free move depending on the arrangement
        if the top layer is filled, include the (self.height-1)th layer (1 indexed)
        else, go up to (self.height-2), 1 indexed
        """
        layer_types = self.layer_type_count()
        return layer_types[1] + 2*layer_types[3]

    def layer_type_count(self):
        """
        returns number of each layer type, not including layers that we cannot remove blocks from

        types are
            one block
            two block (| || |)
            two block (| |   | |)
            three block
        """
        layer_counts = [0,  # one block layers
                        0,  # two block (| || |) layers
                        0,  # two block (| |   | |) layers
                        0,  # three block layers
                        ]
        for L in range(self.height() - 2 + self.top_layer_filled()):
            # if top layer is filled, go one more
            count = self.blocks_on_level(L)
            if count == 1:
                layer_counts[0] += 1
            elif count == 3:
                layer_counts[3] += 1
            else:
                # there are two blocks
                if self.block_info[L][1] is None:
                    # if there is no middle block
                    layer_counts[2] += 1
                else:
                    layer_counts[1] += 1
        return layer_counts

    def height(self):
        """
        :return: height of tower
        """
        return len(self.block_info)

    def com(self):
        """
        center of mass of tower
        :return: np array
        """
        return self.COMs[0]

    def top_layer_filled(self):
        """
        returns if top layer is filled
        """
        return self.blocks_on_level(self.height() - 1) == 3

    def image_of_layer(self, L, resolution=256, radius=None):
        """
        returns an 'image' of density of a layer
        array of 0s or 1s, 1 if there is a block at a point

        :param L: layer of tower
        :param resolution: granularity of the image
        :param radius: distance to look on each dimension
            looks at [-radius,radius]^2

        returns an (resolution x resolution) numpy array 'image'
        """
        if radius is None:
            radius = 1.2*Jenga.TOWER_DIAMETER/2
        grid = (np.arange(resolution)/(resolution - 1) - .5)*2*radius
        xv, yv = np.meshgrid(grid, grid, indexing='xy')
        points_arr = np.stack((xv, yv), axis=2)
        points_arr = points_arr.reshape((resolution*resolution, 2))

        thingies = np.sum(
            [which_points_in_hull(points_arr, hull, tolerance=TOLERANCE) for hull in self.get_block_hulls()[L]], axis=0)

        return thingies.reshape((resolution, resolution))

    # FALLING/TERMINAL CHECKS

    def falls_at_layer(self, L):
        """
        computes whether tower falls at layer L
        :param L: layer of tower (must be < self.height-1)
        :return: boolean of whether the tower above layer L has COM outside of the convex hull of layer L when projected to xy plane
        """
        com = self.COMs[L + 1][:2]  # COM ABOVE level, project to xy

        hull = self.hulls[L]

        # find_simplex returns 0 if point is inside simplex, and -1 if outside.
        # return if it 'falls' i.e. if hull.find_simplex < 0
        return not point_in_hull(com, hull)

    def log_prob_stable(self, scale=.001):
        """
        computes log of probability that tower is stable
            sum across layers
                (equivalent to just multiplying the probabilities)
            could also try min across layers
                (equivalent to probability that the least stable layer is stable)
        :return: log(prob)
        """
        return sum(self.log_prob_stable_at_layer(L, scale=scale) for L in range(self.height() - 1))

    def log_prob_stable_at_layer(self, L, scale=.0005):
        """
        computes log of probability that tower is stable at layer L
            probability is calculated by sigmoid of signed distance from the convex hull
            scaled by some value so that
        :param L: layer of tower (must be < self.height-1)
        :return: log(prob)
        """
        return -np.log(1 + np.exp(self.raw_score_at_layer(L)/scale))

    def raw_score_at_layer(self, L):
        """
        computes the signed distance from COM at layer L+1 to the convex hull of layer L. Can be thought of as an un-normalized score
        :param L: layer of tower (must be < self.height-1)
        :return: -1 * the signed distance from the projection of the convex hull at layer L+1 to the convex hull of layer L
        """
        hull = self.hulls[L]
        com = self.COMs[L + 1][:2]
        return hull_score(com, hull)

    def deterministic_falls(self):
        """
        returns if the tower falls at any level
        """
        return any(self.falls_at_layer(L) for L in range(self.height() - 1))

    def terminal_state(self):
        """
        returns if the tower is at a terminal state (ON THE START OF A TURN, i.e. player about to remove a block)
            this is true if either the tower falls or there are no moves remaining
        """
        return self.deterministic_falls() or len(self.valid_removes()) == 0

    # VALIDITY CHECKS

    def valid_removes(self):
        """
        returns list of all (L,i) pairs that are allowed to be removed
        """
        out = []
        for L in range(self.height() - 2 + self.top_layer_filled()):
            # normally, we cannot remove blocks on the top two layers
            # if top layer is filled, we can remvoe from the layer below it
            if self.blocks_on_level(L) > 1:
                out += [(L, i) for (i, t) in enumerate(self.block_info[L]) if t is not None]
        return out

    def is_valid_remove(self, remove):
        """
        returns is specified block is allowed to be removed
        :param remove: (L,i) tuple
            L: level of block
            i: index of block
        :return: boolean
        """
        L, i = remove
        # initial checks
        if L >= self.height() - 2:
            if not self.top_layer_filled():
                return False
            elif L >= self.height() - 1:
                return False

        # if level will be empty
        if self.blocks_on_level(L) <= 1:
            return False

        # if block does not exist
        if self.block_info[L][i] is None:
            return False
        return True

    def valid_place_blocks(self):
        """
        returns the valid 'moves' to place a block on tower
        :return: non-empty list of indices
        """
        if self.top_layer_filled():
            return [i for i in range(3)]
        return [i for i in range(3) if self.block_info[-1][i] is None]

    def valid_moves_product(self):
        """
        returns all valid next moves
        :return: (all possible 'remove' steps, all possible 'place' steps)
            Note that any choice from these is a valid next move
        """
        removes = self.valid_removes()
        # Note: removing a block does not change the top layer
        # thus, the possible placement moves remain constant after removing a block
        places = self.valid_place_blocks()
        return (removes, places)

    def valid_moves(self):
        moves = []
        removes, places = self.valid_moves_product()
        for remove in removes:
            for place in places:
                moves.append((remove, place))
        return moves

    def has_valid_moves(self):
        """
        returns whether we have any valid moves
        enough to check if there are any valid removes
        """
        return len(self.valid_removes()) > 0

    # GAME MOVE METHODS (RETURNS MUTATED TOWER)

    def remove_block(self, remove):
        """
        removes specified block
        :param remove: (L,i) tuple
            L: level of block
            i: index of block
        :return: Tower object with specified block removed
        """
        L, i = remove
        if not self.is_valid_remove(remove=remove):
            if L >= self.height() - 2:
                if not self.top_layer_filled():
                    print("WARNING: CANNOT REMOVE BLOCK BELOW INCOMPLETE TOP LAYER")
                elif L >= self.height() - 1:
                    print("WARNING: CANNOT REMOVE BLOCK ON TOP LAYER")
            else:
                print("WARNING: BLOCK '" + str(i) + "' INVALID TO REMOVE ON LAYER" + str(
                    [(t is not None) for t in self.block_info[L]]))

        return Jenga(
            [
                [(None if eye == i and ell == L else block) for (eye, block) in enumerate(level)]
                for (ell, level) in enumerate(self.block_info)],
            pos_std=self.pos_std,
            angle_std=self.angle_std,
        )

    def place_block(self, i, blk_pos_std=None, blk_angle_std=None):
        """
        places block at specified position
        :param i: position to add
        :param blk_pos_std: pos stdev, if different from default
        :param blk_angle_std: angle stdev, if different from default
        :return: Tower with block added
        """
        if blk_pos_std is None:
            blk_pos_std = self.pos_std
        if blk_angle_std is None:
            blk_angle_std = self.angle_std
        if i not in self.valid_place_blocks():
            raise Exception(
                "i=" + str(i) + " DOES NOT FIT IN LEVEL " + str([(t is not None) for t in self.block_info[-1]]))

        if self.top_layer_filled():
            new_block = random_block(L=self.height(), i=i, pos_std=blk_pos_std, angle_std=blk_angle_std)
            return Jenga(self.block_info + [[(new_block if eye == i else None) for eye in range(3)]],
                         pos_std=self.pos_std,
                         angle_std=self.angle_std,
                         )
        else:
            new_block = random_block(L=self.height() - 1, i=i, pos_std=blk_pos_std, angle_std=blk_angle_std)
            return Jenga(
                [
                    [(new_block if eye == i and L == self.height() - 1 else block) for (eye, block) in enumerate(level)]
                    for (L, level) in enumerate(self.block_info)],
                pos_std=self.pos_std,
                angle_std=self.angle_std,
            )

    def play_move_deterministic(self, remove, place):
        """
        :param remove: (L,i) tuple, remove ith block from Lth level
        :param place: index of place action
        :return: (Tower object with specified action taken, Boolean for whether tower fell)
        note: remove must be in removes and place in places for (removes,places)=self.valid_moves()
        """
        removes, places = self.valid_moves_product()
        if not (remove in removes and place in places):
            raise Exception("NOT VALID MOVE: " + str(remove) + ',' + str(place))
        removed = self.remove_block(remove)
        placed = removed.place_block(place)
        return placed, (removed.deterministic_falls() or placed.deterministic_falls())

    def play_move_log_probabilistic(self, remove, place):
        """
        :param remove: (L,i) tuple, remove ith block from Lth level
        :param place: index of place action
        :return: (Tower object with specified action taken, log prob of tower being stable)
        note: remove must be in removes and place in places for (removes,places)=self.valid_moves()
        """
        removes, places = self.valid_moves_product()
        if not (remove in removes and place in places):
            raise Exception("NOT VALID MOVE: " + str(remove) + ',' + str(place))
        removed = self.remove_block(remove)
        placed = removed.place_block(place)

        return placed, (removed.log_prob_stable() + placed.log_prob_stable())

    # SAVING/OTHER

    def to_array(self):
        """
        returns tower representation as a list of numpy vectors
        each row is an index (3L+i) along with an encoded block
        """
        # boolean = np.array([[t is not None for t in layer] for layer in self.block_info])
        block_info = np.concatenate([
            np.array([
                np.concatenate(([L*3 + i], t.to_vector()))
                for (i, t) in enumerate(layer) if t is not None])
            for (L, layer) in enumerate(self.block_info)], axis=0)
        return block_info

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CLASS METHODS                                                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

    def valid_special_moves(self):
        """
        returns iterable of special moves possible from current position
        MUST BE DETERMINISTIC, always return moves in same order
        Returns: boolean
        """
        if not self.special_moves:
            return iter(())
        else:
            raise NotImplementedError

    @property
    def observation(self):
        """
        Returns: (board, position, info vector), as observed by the current player
            of shapes ((D1,...,DN, *1),(D1,...,DN, *2),...), (D1,...,DN, N), (T,))
        should return clones of any internal variables

        This can involve flipping the board and such, if necessary
        note that indices may also need to be flipped
            i.e. the opponent's 0 may be different
        """
        raise NotImplementedError

    @staticmethod
    def num_observation_boards():
        """
        Returns: number of boards in (D1,...,DN, *1),(D1,...,DN, *2),...)
        """
        raise NotImplementedError

    @staticmethod
    def underlying_set_sizes():
        """
        returns number of possible distinct elements of each underlying set, if finite
        """
        raise NotImplementedError

    @property
    def representation(self):
        """
        Returns: representation of self, likely a tuple of tensors
            often this is the same as self.observation, (i.e. for perfect info games)
        all information required to play the game must be obtainable from representation
        i.e. self.from_represnetation(self.represnetation) must be functionally equivalent to self

        should return clones of any internal variables
        """
        return self.block_info, self.num_players, self.current_player

    @staticmethod
    def from_representation(representation):
        """
        returns a SubsetGame instance from the output of self.get_representation
        Args:
            representation: output of self.get_representation
        Returns: SubsetGame object
        """
        block_info, num_players, current_player = representation
        return Jenga(block_info=block_info,
                     num_players=num_players,
                     current_player=current_player,
                     )

    def make_move(self, local_move):
        """
        gets resulting SubsetGame object of taking specified move from this state
        this may not be deterministic,
        cannot be called on terminal states
        Args:
            local_move: a subset of the possible obs board indices, a tuple of N-tuples
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

    def __eq__(self, other):
        return self.block_info == other.block_info

    def __str__(self):
        """
        returns string representation of tower
        binary encoding of each level
        i.e. full layer is a 7, layer with one block is a 1, 2, or 4 depending on which block
        blocks are indexed from -x to +x (even layers) or -y to +y (odd layers)
        """
        s = ''
        for L in self.block_info:
            s_L = 0
            for (i, t) in enumerate(L):
                if t is not None:
                    s_L += int(2**i)
            s += str(s_L)
        return s


def tower_from_array(arr):
    """
    returns the tower encoded by array

    :param arr: is output of Tower.to_array
    """
    if arr is None:
        return None
    height = round(arr[-1][0])//3 + 1
    # height is 1 + level of last block
    block_info = [[None for _ in range(3)] for _ in range(height)]
    for vec in arr:
        Li = round(vec[0])
        L, i = Li//3, Li%3
        block_info[L][i] = block_from_vector(vec[1:])
    return Jenga(block_info=block_info)


if __name__ == "__main__":
    b = random_block(1, 1, pos_std=0.)
    t = Jenga(pos_std=0.001, angle_std=0.005)
    from PIL import Image

    layer0 = t.image_of_layer(0, resolution=512)
    img = Image.fromarray(layer0*255)
    img.show()
    t = t.remove_block((Jenga.INITIAL_SIZE - 2, 2))

    layer = t.image_of_layer(Jenga.INITIAL_SIZE - 2, resolution=512)
    img = Image.fromarray(layer*255)
    img.show()
    print(t)

    print(t.deterministic_falls())
    print([(t.raw_score_at_layer(i)) for i in range(len(t.block_info) - 1)])
    t = t.remove_block((Jenga.INITIAL_SIZE - 2, 1))
    print([(t.raw_score_at_layer(i)) for i in range(len(t.block_info) - 1)])
    print(t.deterministic_falls())
    t: Jenga
    t = t.place_block(0)
    t = t.place_block(1)
    t = t.place_block(2)

    print(t.deterministic_falls())
    arr = t.to_array()
    t2 = tower_from_array(arr)
    print('equality:', t == t2)
    print(t.com())
    print(t.top_layer_filled())
    print(t.valid_place_blocks())
