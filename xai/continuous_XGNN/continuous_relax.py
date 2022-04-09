from audioop import add
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from typing import Optional

from scipy.sparse import coo_matrix
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj
import copy
from math import sqrt, floor
from inspect import signature

from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx, sort_edge_index, dense_to_sparse, to_dense_adj
from ..XGNN.gnn_explain import gnn_explain
from ismember import ismember
import os

    
EPS = 1e-15

class GNNExplainer(torch.nn.Module):
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, lr: float = 0.01,
                 num_hops: Optional[int] = None, 
                 log: bool = True, **kwargs):
        super().__init__()
        self.model = model
        self.model_p = copy.deepcopy(model)
        self.lr = lr
        self.__num_hops__ = num_hops
        self.log = log
        self.coeffs.update(kwargs)

    def __set_masks__(self, x, edge_index, perturbed_edge_index, init="normal"):
        (N, F) = x.size()
        E, E_p = edge_index.size(1), perturbed_edge_index.size(1)
        
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        self.perturbed_mask = torch.nn.Parameter(torch.randn(E_p) * std)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

        for module in self.model_p.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.perturbed_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        for module in self.model_p.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None
        self.perturbed_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __loss__(self, pred, pred_perturb):
        
        loss = torch.norm(pred - pred_perturb, 1)
        
        return loss

    def explain_graph(self, x, edge_index, perturbed_edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.
        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.model_p.eval()
        self.__clear_masks__()

        self.__set_masks__(x, edge_index, perturbed_edge_index)
        self.to(x.device)
        
        optimizer = torch.optim.Adam([self.edge_mask, self.perturbed_mask], lr=self.lr)

        for e in range(0, 10):
            #print('gnn_explainer: ' + str(e))
            optimizer.zero_grad()
            out = self.model(x=x, edge_index=edge_index, **kwargs)
            out_p = self.model_p(x=x, edge_index=perturbed_edge_index, **kwargs)
          
            loss = self.__loss__(out, out_p)
            loss.backward()
            optimizer.step()

        edge_mask = self.edge_mask.detach()
        perturbed_mask = self.perturbed_mask.detach()
        self.__clear_masks__()
        return edge_mask, perturbed_mask
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'


class continuous_gnn_explain(gnn_explain):
    
    def add_drop_edge_random(self, add_prob=0.5, del_prob=0.5):
        
        N, F = self.features.size()
        print(self.edge_index)
        E = self.edge_index.size(1)

        # Generate scores 
        scores = torch.Tensor(np.random.uniform(0, 1, (N, N)))

        # DELETE
        
        edits_to_make = floor(E * del_prob)
        top_delete = torch.topk(scores.flatten(), edits_to_make).indices
        base_end = torch.remainder(top_delete, N)
        base_start = torch.floor_divide(top_delete, N)
        end = torch.cat((base_end, base_start))
        start = torch.cat((base_start, base_end))
        delete_indices = torch.stack([end, start])

        # ADD
        edits_to_make = floor(N**2 * add_prob)
        top_adds = torch.topk(scores.flatten(), edits_to_make).indices
        base_end = torch.remainder(top_adds, N)
        base_start = torch.floor_divide(top_adds, N)
        end = torch.cat((base_end, base_start))
        start = torch.cat((base_start, base_end))
        add_indices = torch.stack([end, start])
        
        return delete_indices, add_indices


    def perturb_graph(self, deleted_edges, add_edges):

        # Edges deleted from original edge_index
        delete_indices = []
        self.perturbed_edge_index = copy.deepcopy(self.edge_index)
        print(deleted_edges)
        for edge in deleted_edges.T:
            vals = (self.edge_index == torch.tensor([[edge[0]], [edge[1]]]))
            sum = torch.sum(vals, dim=0)
            col_idx = np.where(sum == 2)[0][0]
            delete_indices.append(col_idx)

        delete_indices.sort(reverse=True)
        for col_idx in delete_indices:
            self.perturbed_edge_index = torch.cat((self.edge_index[:, :col_idx], self.edge_index[:, col_idx+1:]), axis=1)

        # edges added to perturbed edge_index
        start_edges = self.perturbed_edge_index.shape[1]
        add_indices = [i for i in range(start_edges, start_edges + add_edges.shape[1], 1)]
        self.perturbed_edge_index = torch.cat((self.perturbed_edge_index, add_edges), axis=1)

        return delete_indices, add_indices

    def train(self, model_checkpoints_dir, model_file):
        
        self.graph_reset()

        checkpoint = torch.load(os.path.join(model_checkpoints_dir, model_file))
        self.gnnNets.load_state_dict(checkpoint['net'])

        X, A = self.read_from_graph(self.graph)
        self.features = torch.from_numpy(X)
        self.edge_index = torch.from_numpy(A)
        grad_gen = GNNExplainer(self.gnnNets)
    
        print(grad_gen)
        # perturb graph (return the ACTUAL edges)
        deleted_edges, added_edges = self.add_drop_edge_random()         
        # get indices of pertubations in edge list (indices in particular edge_lists)             
        del_indices, add_indices = self.perturb_graph(deleted_edges, added_edges) 
        # generate gradients on these perturbations
        edge_mask, perturbed_mask = grad_gen.explain_graph(self.features, self.edge_index, self.perturbed_edge_index)
        added_grads = perturbed_mask[add_indices]
        deleted_grads = edge_mask[del_indices]
     
        # figure out which perturbations were best 
        best_add_score = torch.min(added_grads)
        best_add_idx = torch.argmin(added_grads)
        best_add = added_edges[:, best_add_idx]

        best_delete_score = torch.min(deleted_grads)
        best_delete_idx = torch.argmin(deleted_grads)
        best_delete = deleted_edges[:, best_delete_idx]
        
        # we want to add edge since better
        if best_add_score < best_delete_score:
            # add both directions since undirected graph
            best_add_comp = torch.tensor([[best_add[1]], [best_add[0]]])
            self.edge_index = torch.cat((self.edge_index, best_add.view(2, 1), best_add_comp), axis=1)
        else: # delete
            val_del = (self.edge_index == torch.tensor([[best_delete[1]], [best_delete[0]]]))
            sum_del = torch.sum(val_del, dim=0)
            col_idx_del = np.where(sum_del == 2)[0][0]
            self.edge_index = torch.cat((self.edge_index[:, :col_idx_del], self.edge_index[:, col_idx_del+1:]), axis=1)

            best_delete_comp = torch.tensor([[best_delete[1]], [best_delete[0]]])
            val_del_comp = (self.edge_index == torch.tensor([[best_delete_comp[1]], [best_delete_comp[0]]]))
            sum_del_comp = torch.sum(val_del_comp, dim=0)
            col_idx_del_comp = np.where(sum_del_comp == 2)[0][0]
            self.edge_index = torch.cat((self.edge_index[:, :col_idx_del_comp], self.edge_index[:, col_idx_del_comp+1:]), axis=1)


