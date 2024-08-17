from torch import nn

from aleph0.networks.architect.beginning.input_embedding import InputEmbedding
from aleph0.networks.architect.middle.former import Former
from aleph0.networks.architect.end.policy_value import PolicyValue


class Architect(nn.Module):
    """
    network architecture
    takes input of SelectionGame.observation (board list, positions, vector) and
        possible selection moves and special moves
    outputs a policy distribution and a value estimate
    """

    def __init__(self,
                 input_embedder: InputEmbedding,
                 former: Former,
                 policy_val: PolicyValue,
                 special_moves,
                 ):
        """
        Args:
            input_embedder: InputEmbedding object to embed input
            former: transformer or cisformer to transform embedded input
            policy_val: goes from former output to policy,value
            special_moves: list of all possible special moves in game
                used to convert from indices to moves and back
        """
        super().__init__()
        self.input_embedder = input_embedder
        self.former = former
        self.policy_val = policy_val
        self.special_moves = special_moves
        self.special_move_to_idx = {move: i
                                    for i, move in enumerate(self.special_moves)}

    def forward(self,
                observation,
                selection_moves,
                special_moves,
                softmax=False,
                ):
        """
        Args:
            observation: (boards, indexes, vector) observation
            selection_moves: selection mvoes, iterable of list(multidim index)
                usable by policy value net
            special_move_idxs: indexes of special moves, usable by policy value net
            softmax: whether to softmax output
        Returns:
            policy, value, same as policy value net
        """
        embedding = self.input_embedder.forward(observation=observation)
        embedding, cls_embedding = self.former.forward(embedding)
        return self.policy_val.forward(embedding=embedding,
                                       cls_embedding=cls_embedding,
                                       selection_moves=selection_moves,
                                       special_move_idxs=[self.special_move_to_idx[move]
                                                          for move in special_moves],
                                       softmax=softmax,
                                       )
