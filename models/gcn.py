import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, depth=2, dropout=0.3):
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))

        for _ in range(depth - 2):
            self.convs.append(
                GCNConv(nhid, nhid))

        self.convs.append(GCNConv(nhid, nclass))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

if __name__ == '__main__':
    
    gcn = GCN(nfeat=2, nhid=8, nclass=1, depth=2, dropout=0.3)
    print(gcn)
