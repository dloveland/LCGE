import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.sage_conv import SAGEConv


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, depth=2, dropout=0.3):
        super(SAGE, self).__init__()
        self.conv_start = SAGEConv(nfeat, nhid, normalize=True)
        self.conv_start.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )

        self.convs_mid = nn.ModuleList()
        self.trans = nn.ModuleList()
        for _ in range(depth - 2):
            conv = SAGEConv(nhid, nhid, normalize=True)
            conv.aggr = 'mean'
            transition = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(nhid),
                nn.Dropout(p=dropout)
            )
            self.convs_mid.append(conv)
            self.trans.append(transition)

        self.conv_end = SAGEConv(nhid, nhid, normalize=True)
        self.conv_end.aggr = 'mean'
        self.lin = nn.Linear(nhid, nclass)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for trans in self.trans:
            trans.reset_parameters() 

    def forward(self, x, edge_index):
        x = self.conv_start(x, edge_index)
        x = self.transition(x)

        for i, conv in enumerate(self.convs_mid):
            x = conv(x, edge_index)
            x = self.trans[i](x)

        x = self.conv_end(x, edge_index)
        x = self.lin(x)
        return x

if __name__ == '__main__':

    sage = SAGE(nfeat=2, nhid=8, nclass=1, depth=2, dropout=0.3)
    print(sage)
