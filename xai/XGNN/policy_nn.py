import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from utils.utils import weights_init
import sys 
sys.path.append('/shared-datadrive/shared-training/LCGE')
from models.gcn import GCN
import torch.nn.functional as F

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    #print("===Normalizing adjacency matrix symmetrically===")
    adj = adj.numpy()
    N = adj.shape[0]
    D = np.sum(adj, 0)
    D_hat = np.diag(np.power(D,-0.5))
    out = np.dot(D_hat, adj).dot(D_hat)
    out[np.isnan(out)]=0
    out = torch.from_numpy(out)
    return out, out.float()




class PolicyNN(nn.Module):
    def __init__(self,  input_dim, node_type_num, initial_dim=8, latent_dim=[16, 24, 32],  max_node = 12): 
        print('Initializing Policy Nets')
        super(PolicyNN, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim  = input_dim
        self.node_type_num =node_type_num 
        self.initial_dim = initial_dim
        self.start_mlp_hidden = 16
        self.tail_mlp_hidden = 24

        self.input_mlp = nn.Linear(self.input_dim, initial_dim)


        self.gcns = nn.ModuleList()
        self.layer_num = len(latent_dim)
        self.gcns.append(GCN(self.initial_dim, self.latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.gcns.append(GCN(self.latent_dim[i-1], self.latent_dim[i]))
        
        self.dense_dim = latent_dim[-1]

        self.start_mlp1= nn.Linear(self.dense_dim, self.start_mlp_hidden)
        self.start_mlp_non_linear = nn.ReLU6()
        self.start_mlp2= nn.Linear(self.start_mlp_hidden, 1)


        self.tail_mlp1= nn.Linear(2*self.dense_dim, self.tail_mlp_hidden)
        self.tail_mlp_non_linear = nn.ReLU6()
        self.tail_mlp2= nn.Linear(self.tail_mlp_hidden, 1)

        # starting from here are the layers added for adding deletion
        self.add_delete_mlp_1 = nn.Linear(1, 1) # > 0.5 delete, <0.5 add
        self.delete_start_mlp1 = nn.Linear(self.dense_dim, self.start_mlp_hidden)
        self.delete_start_mlp_non_linear = nn.ReLU6()
        self.delete_start_mlp2= nn.Linear(self.start_mlp_hidden, 1)

        weights_init(self)


    def forward(self, node_feat, n2n_sp, node_num):

        un_A, A = normalize_adj(n2n_sp)
        cur_out = node_feat
        cur_A = A

        cur_out = self.input_mlp(cur_out)

        for i in range(self.layer_num):
            cur_out = self.gcns[i](cur_A, cur_out)
        
        ### now we have the node embeddings
        ### pass through an MLP to determine whether to do add or deletion
        graph_embedding = torch.mean(cur_out)
        graph_embedding = graph_embedding.unsqueeze(0)
        add_or_delete = self.add_delete_mlp_1(graph_embedding)
        add_or_delete = torch.sigmoid(add_or_delete)[0]

        if add_or_delete <= 0.5: # add

            """  the following code are for addition """

            ### get two different masks
            ob_len = node_num ##total current + candidates set
            ob_len_first = ob_len - self.node_type_num
            
            logits_mask = self.sequence_mask(ob_len, cur_A.size()[0])
            
            logits_mask_first = self.sequence_mask(ob_len_first, cur_A.size()[0])


            ### action--- select the starting node, two layer mlps
            
            start_emb = self.start_mlp1(cur_out)
            start_emb = self.start_mlp_non_linear(start_emb)
            start_logits = self.start_mlp2(start_emb)
            start_logits_ori = torch.squeeze(start_logits)
            start_logits_short = start_logits_ori[0:ob_len_first]
            
            start_probs = torch.nn.functional.softmax(start_logits_short,dim=0)
            
                
            start_prob_dist = torch.distributions.Categorical(start_probs)
            try:
                start_action = start_prob_dist.sample()
            except:
                import pdb
                pdb.set_trace()
            

            mask = F.one_hot(start_action, num_classes=node_feat.size()[0])
            mask = mask.bool()
            emb_selected_node = torch.masked_select(cur_out,mask[:,None]) 

            ### action--- select the tail node, two layer mlps
            emb_selected_node_copy = emb_selected_node.repeat(cur_out.size()[0],1)
            cat_emb = torch.cat((cur_out, emb_selected_node_copy), 1)

            tail_emb = self.tail_mlp1(cat_emb)
            tail_emb = self.tail_mlp_non_linear(tail_emb)
            tail_logits= self.tail_mlp2(tail_emb)
            tail_logits_ori =torch.squeeze(tail_logits)


            logits_second_mask = logits_mask[0] & ~mask
            tail_logits_short =  tail_logits_ori[0:ob_len] 
            logits_second_mask_short = logits_second_mask[0:ob_len]
            
            
            tail_logits_null = torch.ones_like(tail_logits_short)*-1000000 
            tail_logits_short = torch.where(logits_second_mask_short==True, tail_logits_short, tail_logits_null)
            
            tail_probs = torch.nn.functional.softmax(tail_logits_short,dim=0)
            

            
            tail_prob_dist = torch.distributions.Categorical(tail_probs)
            try:
                tail_action = tail_prob_dist.sample()
            except:
                import pdb
                pdb.set_trace()

        else: # delete
            """  the following code are for deletion """

            ### get two different masks
            ob_len = node_num ##total current + candidates set
            ob_len_first = ob_len - self.node_type_num
            
            logits_mask = self.sequence_mask(ob_len, cur_A.size()[0])
            
            logits_mask_first = self.sequence_mask(ob_len_first, cur_A.size()[0])
            ### action--- select the starting node, two layer mlps

            start_emb = self.delete_start_mlp1(cur_out)
            start_emb = self.delete_start_mlp_non_linear(start_emb)
            start_logits = self.delete_start_mlp2(start_emb)
            start_logits_ori = torch.squeeze(start_logits)
            start_logits_short = start_logits_ori[0:ob_len_first]
            
            start_probs = torch.nn.functional.softmax(start_logits_short,dim=0)
            
                
            start_prob_dist = torch.distributions.Categorical(start_probs)
            try:
                start_action = start_prob_dist.sample()
            except:
                import pdb
                pdb.set_trace()
            

            mask = F.one_hot(start_action, num_classes=node_feat.size()[0])
            # mask out the one-hop neighbors of the start node
            one_hop_neighbor_mask = ~(A[start_action]==0)
            mask = mask.bool()
            # deletion should be chosen from one-hop neighbors
            # use the start node embedding as node features
            emb_selected_node = torch.masked_select(cur_out,mask[:,None]) 

            ### action--- select the tail node, two layer mlps
            emb_selected_node_copy = emb_selected_node.repeat(cur_out.size()[0],1)
            cat_emb = torch.cat((cur_out, emb_selected_node_copy), 1)

            tail_emb = self.tail_mlp1(cat_emb)
            tail_emb = self.tail_mlp_non_linear(tail_emb)
            tail_logits= self.tail_mlp2(tail_emb)
            tail_logits_ori =torch.squeeze(tail_logits)


            logits_second_mask = logits_mask[0] & ~mask
            logits_second_mask = logits_second_mask & one_hop_neighbor_mask
            tail_logits_short =  tail_logits_ori[0:ob_len] 
            logits_second_mask_short = logits_second_mask[0:ob_len]
            
            
            tail_logits_null = torch.ones_like(tail_logits_short)*-1000000 
            tail_logits_short = torch.where(logits_second_mask_short==True, tail_logits_short, tail_logits_null)

            tail_probs = torch.nn.functional.softmax(tail_logits_short,dim=0)
            

            
            tail_prob_dist = torch.distributions.Categorical(tail_probs)
            try:
                tail_action = tail_prob_dist.sample()
            except:
                import pdb
                pdb.set_trace()            

        return add_or_delete, start_action, start_logits_ori, tail_action, tail_logits_ori 


    def sequence_mask(self, lengths, maxlen, dtype=torch.bool):
        mask = ~(torch.ones((lengths, maxlen)).cumsum(dim=1).t() > lengths).t()
        mask.type(dtype)
        return mask