from torch_geometric.utils import from_networkx
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import networkx as nx
import matplotlib
# import datasets
matplotlib.use('agg')


def draw_graph(G, filename="graph.png"):
    nx.draw(G)
    plt.savefig(filename)
    plt.clf()



def format_data(G, is_cyclic=0):
    data = from_networkx(G)
    node_degree = torch.zeros(G.number_of_nodes(), 1)
    for idx in np.arange(data.edge_index.shape[1]):
        node_degree[data.edge_index[0, idx]-1] += 1
    # max_node_degree = torch.max(node_degree)
    # if max_node_degree > datasets.MAX_DEGREE:
    #     datasets.MAX_DEGREE = max_node_degree
    data.x = node_degree
    data.y = is_cyclic
    return data


def generate_acyclic():
    num_graphs_per_type = 10

    # 1. Generate cyclic graphs
    cyclic_graphs = []
    is_cyclic = 1

    # cycles
    for num_nodes in range(3, 3 + num_graphs_per_type):
        G = nx.cycle_graph(num_nodes)
        cyclic_graphs.append(format_data(G, is_cyclic))

    # grids
    permutation_1 = np.random.permutation(np.arange(2, 2 + num_graphs_per_type))
    permutation_2 = np.random.permutation(np.arange(2, 2 + num_graphs_per_type))
    for idx in range(num_graphs_per_type):
        G = nx.grid_2d_graph(permutation_1[idx], permutation_2[idx])
        cyclic_graphs.append(format_data(G, is_cyclic))
        # draw_graph(G, f"graph_{idx}.png")

    # wheels
    for num_nodes in range(4, 4 + num_graphs_per_type):
        G = nx.wheel_graph(num_nodes)
        cyclic_graphs.append(format_data(G, is_cyclic))
        # draw_graph(G, f"wheel_{num_nodes}.png")

    # circular ladders
    for num_nodes in range(3, 3 + num_graphs_per_type):
        G = nx.circular_ladder_graph(num_nodes)
        cyclic_graphs.append(format_data(G, is_cyclic))
        # draw_graph(G, f"circular_ladder_{num_nodes}.png")

    # 2. Generate acyclic graphs
    acyclic_graphs = []
    is_cyclic = 0

    # # stars
    for num_nodes in range(3, 3 + num_graphs_per_type):
        G = nx.star_graph(num_nodes)
        acyclic_graphs.append(format_data(G, is_cyclic))
        # draw_graph(G, f"star_{num_nodes}.png")

    # balanced binary trees
    for num_nodes in range(1, 1 + num_graphs_per_type):
        G = nx.balanced_tree(2, num_nodes)
        acyclic_graphs.append(format_data(G, is_cyclic))
        # draw_graph(G, f"binary_tree_{num_nodes}.png")

    # path
    for num_nodes in range(4, 4 + num_graphs_per_type):
        G = nx.path_graph(num_nodes)
        acyclic_graphs.append(format_data(G, is_cyclic))
        # draw_graph(G, f"path_{num_nodes}.png")

    # full rary trees
    for num_nodes in range(7, 7 + 2 * num_graphs_per_type, 2):
        G = nx.full_rary_tree(2, num_nodes)
        acyclic_graphs.append(format_data(G, is_cyclic))
        # draw_graph(G, f"rary_tree_{num_nodes}.png")

    acyclic_dataset = []
    acyclic_dataset.extend(acyclic_graphs)
    acyclic_dataset.extend(cyclic_graphs)

    dataset_path = Path("datasets") / "acyclic"

    with open(dataset_path / "preprocessed_data.pkl", 'wb') as f:
        pickle.dump(acyclic_dataset, f)

    return acyclic_dataset


if __name__ == '__main__':
    generate_acyclic()
