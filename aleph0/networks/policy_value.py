import torch
from torch import nn

from aleph0.networks.ffn import FFN


class PolicyValue(nn.Module):
    def __init__(self,
                 embedding_dim,
                 selection_size,
                 num_special_moves,
                 num_players,
                 policy_hidden_layers=None,
                 value_hidden_layers=None,
                 special_hidden_layers=None,

                 ):
        """
        Args:
            embedding_dim: dimension of embedding input
            selection_size: number of indices to select
            num_special_moves: number of possible special moves to include into policy
            num_players: number of players (determines dimension of value output
            policy_hidden_layers: hidden layers to use for policy net
            value_hidden_layers: hidden layers to use for value net
            special_hidden_layers: guess
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.selection_size = selection_size
        self.selection_policy_net = FFN(input_dim=self.embedding_dim*self.selection_size,
                                        output_dim=1,
                                        hidden_layers=policy_hidden_layers,
                                        )

        self.value_net = FFN(input_dim=embedding_dim,
                             output_dim=num_players,
                             hidden_layers=value_hidden_layers,
                             )
        self.num_special_moves = num_special_moves
        if self.num_special_moves:
            self.special_net = FFN(input_dim=embedding_dim,
                                   output_dim=self.num_special_moves,
                                   hidden_layers=special_hidden_layers,
                                   )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding, cls_embedding, selection_moves, special_move_idxs, softmax=False):
        """
        returns policy,value for the embedding and associated moves
        for the most part batch size is 1, as masking would get very annoying very fast
        Args:
            embedding: (M, D1, ...,DN, E)
            cls_embedding: (M, E)
            selection_moves: will be an iterable of tuples of selection_size indices
                each index will be an N-dim index, corresponding with D1,...,DN
            special_move_idxs: iterable of indices of special move indexes
                assumes we have a global list of special moves that we are indexing
        Returns:
            (M, len(selection_moves)+len(special_move_idxs)) pre-softmax logits
                (M,num_players) value estimates
        """
        M, _ = cls_embedding.shape
        selection_moves = list(selection_moves)
        selections = []
        # do each element of batch independently for now
        for m in range(M):
            sel = torch.zeros(len(selection_moves), self.selection_size*self.embedding_dim)
            # set selections by order
            for k, kth_idxs in enumerate(zip(*selection_moves)):
                # kth_idxs is a list of n-dim indices of size len(selection_moves).
                # we are setting the kth segment of selection to the board index indicated by
                sel[:, k*self.embedding_dim:(k + 1)*self.embedding_dim] = embedding[m][kth_idxs]
            selections.append(sel)
        # size (M, len(selection_moves), self.selection_size*self.embedding_dim)
        selections = torch.stack(selections, dim=0)
        # (M, selection_moves)
        policy = self.selection_policy_net.forward(selections).reshape(M, -1)
        if self.num_special_moves:
            # (M, num_special_moves)
            special_policy = self.special_net.forward(cls_embedding)[:, list(special_move_idxs)]

            policy = torch.cat((policy, special_policy), dim=1)
        # policy is now (M, total moves)

        if softmax:
            policy = self.softmax(policy)

        value = self.value_net.forward(cls_embedding)

        return policy, value
