import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP as APPNP_base

class APPNP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, K=2, alpha=0.1, dropout=0.5):
        super(APPNP, self).__init__()

        self.lins = nn.ModuleList()

        # self.lins.append(nn.Linear(nfeat, nhid))


        # for _ in range(depth - 2):
        #     self.lins.append(nn.Linear(nhid, nhid))
 
        # self.lins.append(nn.Linear(nhid, nclass))
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(nfeat, nhid[0]))
        
        for i in range(1, len(nhid)):
            self.lins.append(nn.Linear(nhid[i-1], nhid[i]))

        self.lins.append(nn.Linear(nhid[-1], nclass))

        self.prop = APPNP_base(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameter() 
        self.prop.reset_parameters()

    def forward(self, x, edge_index):

        for lin in self.lins:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(lin(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop(x, edge_index)
        
        return x


if __name__ == '__main__':

    # appnp = APPNP(nfeat=2, nhid=8, nclass=1, depth=2, K=2, alpha=0.1, dropout=0.5)
    appnp = APPNP(nfeat=2, nhid=[8, 8], nclass=1, K=2, alpha=0.1, dropout=0.5)
    print(appnp)
