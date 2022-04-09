import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from training.gnns import DisNets
from datasets.get_loader import get_loader
from torch_geometric.utils import to_dense_adj 


def train(cfg):
    gnnNets = DisNets(input_dim=cfg.input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gnnNets.parameters(), lr=cfg.gnn_learning_rate)  # cfg.gnn_momentum, cfg.gnn_weight_decay)

    gnnNets.train()
    # print('start loading data')
    # print(os.path)
    # path_adj = os.path.join(cfg.dataset_dir, cfg.adjacency_matrix_file)
    # path_labels = os.path.join(cfg.dataset_dir, cfg.graph_labels_file)
    # path_graph_indicator = os.path.join(cfg.dataset_dir, cfg.graph_indicator_file)
    # path_labels_node = os.path.join(cfg.dataset_dir, cfg.node_labels_file)

    # with open(path_labels_node, 'r') as f:
    #     nodes_all_temp = f.read().splitlines()
    #     nodes_all = [int(i) for i in nodes_all_temp]

    # adj_all = np.zeros((len(nodes_all), len(nodes_all)))

    # with open(path_adj, 'r') as f:
    #     adj_list = f.read().splitlines()

    # for item in adj_list:
    #     lr = item.split(', ')
    #     l = int(lr[0])
    #     r = int(lr[1])
    #     adj_all[l - 1, r - 1] = 1

    # with open(path_graph_indicator, 'r') as f:
    #     graph_indicator_temp = f.read().splitlines()
    #     graph_indicator = [int(i) for i in graph_indicator_temp]
    #     graph_indicator = np.array(graph_indicator)

    # with open(path_labels, 'r') as f:
    #     graph_labels_temp = f.read().splitlines()
    #     graph_labels = [int(i) for i in graph_labels_temp]

    # data_feature = []
    # data_adj = []
    # labels = []

    # for i in range(1, 189):
    #     idx = np.where(graph_indicator == i)
    #     graph_len = len(idx[0])
    #     adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
    #     data_adj.append(adj)
    #     label = graph_labels[i - 1]
    #     labels.append(int(label == 1))
    #     feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
    #     nb_clss = 7
    #     targets = np.array(feature).reshape(-1)
    #     one_hot_feature = np.eye(nb_clss)[targets]
    #     data_feature.append(one_hot_feature)
    #     # print(one_hot_feature)
    #     # print()
        
    # data_size = len(labels)
    # print('The size of dataset is ', data_size)
    
    data_loader = get_loader(cfg.dataset, mode=0, shuffle=True)
    print('The size of dataset is ', len(data_loader.dataset))

    best_acc = 0
    for epoch in range(cfg.gnn_epochs):
        acc = []
        loss_list = []
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            X = torch.tensor(batch.x).squeeze(0)
            A = to_dense_adj(batch.edge_index).squeeze(0).to(torch.float64)  # squeeze if batch_size = 1
            label = batch.y
            
            logits, probs = gnnNets(X.float(), A.float())
            _, prediction = torch.max(logits, 0)
            loss = criterion(logits[None, :], label.long())
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            
            acc.append(prediction.eq(label).item())
            # print(A)
            # print()
            # break
        # orders = np.random.permutation(data_size)
        # for i in range(data_size):
        #     optimizer.zero_grad()
        #     X = data_feature[orders[i]]
        #     A = data_adj[orders[i]]
        #     label = labels[orders[i]:orders[i] + 1]
        #     label = np.array(label)
        #     label = torch.from_numpy(label)
        #     X = torch.from_numpy(X)
        #     A = torch.from_numpy(A)
        #     logits, probs = gnnNets(X.float(), A.float())
        #     _, prediction = torch.max(logits, 0)
        #     loss = criterion(logits[None, :], label.long())
        #     loss_list.append(loss.item())
        #     loss.backward()
        #     optimizer.step()

        #     acc.append(prediction.eq(label).item())
        print("Epoch:%d  |Loss: %.3f | Acc: %.3f" % (epoch, np.average(loss_list), np.average(acc)))

        if (np.average(acc) > best_acc):
            print('saving....')
            state = {
                'net': gnnNets.state_dict(),
                'acc': np.average(acc),
                'epoch': epoch,
            }
            if not os.path.isdir(cfg.save_dir):
                os.makedirs(cfg.save_dir)
            torch.save(state, os.path.join(cfg.model_checkpoints_dir, cfg.model_file))
            best_acc = np.average(acc)
    print('best training accuracy is: ', best_acc)
