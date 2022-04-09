import sys
# sys.path.append('/shared-datadrive/shared-training/LCGE')
from models.gcn import GCN
import torch
import torch.nn as nn
from xai.XGNN.pytorch_util import normalize_adj
from utils.utils import weights_init


class DisNets(nn.Module):
    def __init__(self, input_dim=7, latent_dim=[32, 48, 64], mlp_hidden=32, num_class=2):
        print('Initializing DisNets')
        super(DisNets, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.mlp_hidden = mlp_hidden

        # self.input_mlp = nn.Linear(self.input_dim, initial_dim)

        self.gcns = nn.ModuleList()
        self.layer_num = len(latent_dim)
        self.gcns.append(GCN(input_dim, self.latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.gcns.append(GCN(self.latent_dim[i - 1], self.latent_dim[i]))

        self.dense_dim = latent_dim[-1]

        self.Softmax = nn.Softmax(dim=0)

        if mlp_hidden == -1:
            self.h_weights = nn.Linear(self.dense_dim, num_class)
        else:
            self.h1_weights = nn.Linear(self.dense_dim, mlp_hidden)
            self.h2_weights = nn.Linear(mlp_hidden, num_class)
            self.mlp_non_linear = nn.ELU()

        weights_init(self)

    def forward(self, node_feat, adj_matrix):
        logits, probs = self.embedding(node_feat, adj_matrix)
        return logits, probs

    def embedding(self, node_feat, n2n_sp):
        un_A, A = normalize_adj(n2n_sp)

        cur_out = node_feat
        cur_A = A.float()

        for i in range(self.layer_num):
            cur_out = self.gcns[i](cur_A, cur_out)

        graph_embedding = torch.mean(cur_out, 0)

        if self.mlp_hidden == -1:
            logits = self.h_weights(graph_embedding)
        else:
            h1 = self.h1_weights(graph_embedding)
            h1 = self.mlp_non_linear(h1)
            logits = self.h2_weights(h1)
        probs = self.Softmax(logits)

        return logits, probs