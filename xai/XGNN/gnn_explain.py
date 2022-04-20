import json
import os
from pyexpat import model
import random

import networkx as nx
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim
from training.gnns import DisNets
from xai.continuous_XGNN.utils import progress_bar
from ..XGNN.policy_nn import PolicyNN
from training.gnns import DisNets
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
import sys
from datasets.get_loader import get_loader
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


class gnn_explain():
    def __init__(self, cfg):
        self.graph = nx.Graph()
        # self.mol = Chem.RWMol()  ### keep a mol obj, to check if the graph is valid
        self.max_node = cfg.max_nodes
        self.max_step = cfg.max_steps
        self.max_iters = cfg.max_iters
        self.num_class = cfg.num_classes
        self.node_type = cfg.node_types
        self.learning_rate = cfg.learning_rate
        self.roll_out_alpha = cfg.roll_out_alpha
        self.roll_out_penalty = cfg.roll_out_penalty
        self.policyNets = PolicyNN(self.node_type, self.node_type)
        self.gnnNets = DisNets(cfg.input_dim)
        self.reward_stepwise = cfg.reward_stepwise
        self.target_class = cfg.target_class
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.policyNets.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        node_attributes = json.load(open(cfg.data_file, 'r', encoding='utf-8'))
        self.dict = {int(k): v for k, v in node_attributes["node_types"].items()}
        self.color = {int(k): v for k, v in node_attributes["node_colors"].items()}
        self.max_poss_degree = {int(k): v for k, v in node_attributes["node_max_degree"].items()}
        self.start_from = cfg.start_from
        self.cfg = cfg

    def train(self, model_checkpoints_dir, model_file, test_idx=0, iter_count=0):
        print('Training gnn_explain Started...')
        # given the well-trained model, Load the model
        checkpoint = torch.load(os.path.join(model_checkpoints_dir, model_file))
        self.gnnNets.load_state_dict(checkpoint['net'])
 
        print(self.cfg.model_file)
        base_file = self.cfg.model_file.split('.')[0]
        print(base_file)
        if not os.path.exists('xai_results/{0}'.format(base_file)):
            os.makedirs('xai_results/{0}'.format(base_file))

        param_str = 'max_node_{0}_max_step_{1}_max_iters_{2}'.format(self.max_node, self.max_step, self.max_iters)
        save_folder = 'xai_results/{0}/{1}'.format(base_file, param_str)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
 
        # Get test data loader to use for explanations
        test_data_loader = get_loader(self.cfg.dataset, mode=2, shuffle=False, batch_size=1)
        test_data = []
        #TODO fix this, shouldnt need to unpack dataloader for one datapoint
        for d in test_data_loader: 
            # Just use one for now and handle mutiple explanations in main?
            test_data.append(d)
        test_graph = test_data[test_idx]

        for i in range(self.max_iters):
            if self.start_from == "existing":
                # Set the self.graph to a particular test graph to start the editing process for local explanations
                self.seeded_graph_reset(test_graph.x, test_graph.edge_index)
                self.target_class = test_graph.y
            else:
                # Get the self.graph to a single node graph to start the editing process for global explanations 
                self.graph_reset()

            for j in range(self.max_step):
                self.optimizer.zero_grad()
                reward_pred = 0
                reward_step = 0
                n = self.graph.number_of_nodes()
                if (n > n + self.max_node):
                    break
                self.graph_old = copy.deepcopy(self.graph)
                # get the embeddings
                X, A = self.read_from_graph(self.graph)
                X = torch.from_numpy(X)
                A = torch.from_numpy(A)

                # Feed to the policy nets for actions
                add_or_delete, start_action, start_logits_ori, tail_action, tail_logits_ori = self.policyNets(X.float(), A.float(),
                                                                                            n + self.node_type)

                print(add_or_delete)
                # flag is used to track whether adding/deleting operation is success/valid.
                if tail_action >= n:  # we need to add node, then add edge
                    if n == self.max_node:
                        flag = False
                    elif add_or_delete <= 0.5: # add node & edge
                        print("addition")
                        self.add_node(self.graph, n, tail_action.item() - n)
                        flag = self.add_edge(self.graph, start_action.item(), n)                        
                    else: #  for deleteion, you always need to start from an existing node
                        flag = False
                        print("Both the start and tail node of deletion should exist in the orig graph")
                else: # only add/delete edge
                    if add_or_delete <= 0.5:
                        print("addition")
                        flag = self.add_edge(self.graph, start_action.item(), tail_action.item())
                    else:
                        print("deletion")
                        flag= self.delete_edge(self.graph, start_action.item(), tail_action.item())
                        # check if nodes are isolated, if it is, delete it
                        # if nx.is_isolate(self.graph, start_action.item()):
                        #     self.delete_node(self.graph, start_action.item())
                        # if nx.is_isolate(self.graph, tail_action.item()):
                        #     self.delete_node(self.graph, tail_action.item())

                if flag:
                    validity = self.check_validity(self.graph)
                    # edge added successfully
                    if validity:
                        reward_step = self.reward_stepwise
                        X_new, A_new = self.read_from_graph_raw(self.graph)
                        X_new = torch.from_numpy(X_new)
                        A_new = torch.from_numpy(A_new)
                        logits, probs = self.gnnNets(X_new.float(), A_new.float())
            
                        # based on logits, define the reward
                        _, prediction = torch.max(logits, 0)
                        if self.target_class == prediction:  # positive reward
                            reward_pred = probs[prediction] - 0.5
                        else:  # negative reward
                            reward_pred = probs[self.target_class] - 0.5

                        # Then we need to roll out.
                        reward_rollout = []
                        for roll in range(10):
                            reward_cur = self.roll_out(self.graph, j)
                            reward_rollout.append(reward_cur)
                        reward_avg = torch.mean(torch.stack(reward_rollout))
                        # desgin loss (need to tune the hyper-parameters here)
                        total_reward = reward_step + reward_pred + reward_avg * self.roll_out_alpha

                        if total_reward < 0:
                            self.graph = copy.deepcopy(self.graph_old)  # rollback
                        #  total_reward= reward_step+reward_pred
                        loss = total_reward * (self.criterion(start_logits_ori[None, :], start_action.expand(1))
                                            + self.criterion(tail_logits_ori[None, :], tail_action.expand(1)))
                    else:
                        total_reward = -1  # graph is not valid
                        self.graph = copy.deepcopy(self.graph_old)
                        loss = total_reward * (self.criterion(start_logits_ori[None, :], start_action.expand(1))
                                            + self.criterion(tail_logits_ori[None, :], tail_action.expand(1)))
                else:
                    # in case adding edge was not successful
                    # do not evaluate
                    reward_step = -1
                    total_reward = reward_step + reward_pred
                    # print(start_logits_ori)
                    # print(tail_logits_ori)
                    loss = total_reward * (self.criterion(start_logits_ori[None, :], start_action.expand(1)) +
                                        self.criterion(tail_logits_ori[None, :], tail_action.expand(1)))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                self.optimizer.step()
        # self.graph_draw(self.graph)
        # plt.show()
        # plt.savefig("Graph.png", format="PNG")
        # plt.clf()
        X_new, A_new = self.read_from_graph_raw(self.graph)
        X_new = torch.from_numpy(X_new)
        A_new = torch.from_numpy(A_new)
        logits, probs = self.gnnNets(X_new.float(), A_new.float())
        prob = probs[self.target_class].item()

        # iter_count = 0
        # if os.path.exists('{0}/test_idx_{1}_iter_{2}.npz'.format(save_folder, test_idx, iter_count)):
        #     files = os.listdir('{0}'.format(save_folder))
        #     files.sort()
        #     iter_count = int(files[-1].split('_')[-1].split('.')[0]) + 1
        np.savez('{0}/test_idx_{1}_iter_{2}.npz'.format(save_folder, test_idx, iter_count), \
                 X_new.numpy(), A_new.numpy(), probs.detach().numpy(), self.target_class)

        print("Probability of GNN's target class given the generated pattern: ", prob)

    def graph_draw(self, graph):
        attr = nx.get_node_attributes(graph, "label")
        labels = {}
        # color = ''
        color = []
        for n in attr:
            labels[n] = self.dict[attr[n]]
            # color = color+self.color[attr[n]]
            color.append(self.color[attr[n]])

        nx.draw(graph, labels=labels, node_color=color, with_labels=True)

    def check_validity(self, graph):
        node_types = nx.get_node_attributes(graph, 'label')
        for i in range(graph.number_of_nodes()):
            degree = graph.degree(i)
            max_allow = self.max_poss_degree[node_types[i]]
            if (degree > max_allow):
                return False
        return True

    def roll_out(self, graph, j):
        cur_graph = copy.deepcopy(graph)
        step = 0
        while cur_graph.number_of_nodes() <= self.max_node and step < self.max_step - j:
            #  self.optimizer.zero_grad()
            graph_old = copy.deepcopy(cur_graph)
            step = step + 1
            X, A = self.read_from_graph(cur_graph)
            n = cur_graph.number_of_nodes()
            X = torch.from_numpy(X)
            A = torch.from_numpy(A)
            start_action, start_logits_ori, tail_action, tail_logits_ori = self.policyNets(X.float(), A.float(),
                                                                                           n + self.node_type)
            if tail_action >= n:  # we need add node, then add edge
                if n == self.max_node:
                    flag = False
                else:
                    self.add_node(cur_graph, n, tail_action.item() - n)
                    flag = self.add_edge(cur_graph, start_action.item(), n)
            else:
                flag = self.add_edge(cur_graph, start_action.item(), tail_action.item())

            # if the graph is not valid in rollout, two possible solutions
            # 1. return a negative reward as overall reward for this rollout  --- what we do here.
            # 2. compute the loss but do not update model parameters here--- update with the step loss togehter.
            if flag == True:
                validity = self.check_validity(cur_graph)
                if validity == False:
                    return torch.tensor(self.roll_out_penalty)
            else:  # case 1: add edges but already exists, case2: keep add node when reach max_node
                return torch.tensor(self.roll_out_penalty)

        # Then we evaluate the final graph
        X_new, A_new = self.read_from_graph_raw(cur_graph)
        X_new = torch.from_numpy(X_new)
        A_new = torch.from_numpy(A_new)
        logits, probs = self.gnnNets(X_new.float(), A_new.float())
        reward = probs[self.target_class] - 0.5
        return reward

    def add_node(self, graph, idx, node_type):
        graph.add_node(idx, label=node_type)
        return

    def delete_node(self, graph, idx):
        graph.remove_node(idx)
        return

    def add_edge(self, graph, start_id, tail_id):
        if graph.has_edge(start_id, tail_id) or start_id == tail_id:
            return False
        else:
            graph.add_edge(start_id, tail_id)
            return True

    def delete_edge(self, graph, start_id, tail_id):
        if start_id == tail_id:
            return False
        elif graph.has_edge(start_id, tail_id) is False:
            return False
        else:
            graph.remove_edge(start_id, tail_id)
            return True

    def read_from_graph(self, graph):  # read graph with added candidates nodes
        n = graph.number_of_nodes()
        # degrees = [val for (node, val) in self.graph.degree()]
        F = np.zeros((n + self.max_node + self.node_type, self.node_type))
        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())
        nb_clss = self.node_type
        targets = np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        F[:n, :] = one_hot_feature
        # then get the onehot features for the candidates nodes
        F[n:n + self.node_type, :] = np.eye(self.node_type)

        E = np.zeros([n + self.max_node + self.node_type, n + self.max_node + self.node_type])
        E[:n, :n] = np.asarray(nx.to_numpy_matrix(graph))
        E[:self.max_node + self.node_type, :self.max_node + self.node_type] += np.eye(self.max_node + self.node_type)
        return F, E

    def read_from_graph_raw(self, graph):  # do not add more nodes
        n = graph.number_of_nodes()
        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())
        nb_clss = self.node_type
        targets=np.array(attr).reshape(-1).astype(int)
        one_hot_feature = np.eye(nb_clss)[targets]

        E = np.zeros([n, n])
        E[:n,:n] = np.asarray(nx.to_numpy_matrix(graph))

        return one_hot_feature, E

    def seeded_graph_reset(self, x, edge_index):
        # Takes in a pytorch geometric data and creates an attributed networkx object 
        self.graph.clear()
        edge_index = edge_index.numpy()
        feat_idx = np.argmax(x.numpy(), axis=1)
        for n in range(x.shape[0]):
            self.graph.add_node(n, label=feat_idx[n])

        for e in range(edge_index.shape[1]):
            self.graph.add_edge(edge_index[0][e], edge_index[1][e])
        self.step = 0
        return
        

    def graph_reset(self):
        self.graph.clear()
        # instead of starting from Carbon, start from some random node:
        start_node_attr = random.randint(0, self.node_type - 1)
        self.graph.add_node(0, label=start_node_attr)  # label from self.dict
        self.step = 0
        return
