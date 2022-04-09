import os
import pickle
from pathlib import Path
from platform import node
import random

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import GeometricShapes
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import argparse
from datasets.generate_graph import generate_acyclic

MAX_DEGREE = -1

def num_class_dict(dataset_name):
    dict = {
        "colors": 3,
        "triangles": 11,
        "acyclic": 2,
        "MUTAG": 7,
    }

    return dict[dataset_name]

def dataset_prefix_dict(dataset_name):
    dict = {
        "colors" : "COLORS-3",
        "triangles" : "TRIANGLES",
        "acyclic" : "acyclic",
        "MUTAG" : "MUTAG",
    }
    return dict[dataset_name]

def preprocessing_for_all():
    dataset_names = ["colors", "triangles", "acyclic", "MUTAG"]
    dataset_prefixes = ["COLORS-3", "TRIANGLES", "acyclic", "MUTAG"]
    for dataset_name, dataset_prefix in zip(dataset_names, dataset_prefixes):
        preprocessed_data(dataset_name, dataset_prefix)


def preprocessed_data(dataset_name, file_prefix):
    dataset_path = Path("datasets") / dataset_name
    print(f"Preprocessing dataset {dataset_name}")

    if not os.path.exists(str(dataset_path / "preprocessed_data.pkl")):
        # edges mapping dict: {node_src_idx: [node_dst_idx, ...]}
        if dataset_name == "acyclic":
            graph_data_list = generate_acyclic()
        else:
            with open(dataset_path / f"{file_prefix}_A.txt", 'r') as read_A:
                lines = read_A.read().splitlines()
            A_dict = {}
            for line in lines:
                node_src_idx, node_dst_idx = line.split(", ")
                node_src_idx = int(node_src_idx)
                node_dst_idx = int(node_dst_idx)
                node_dst_idx_lst = A_dict.get(node_src_idx, [])
                node_dst_idx_lst.append(node_dst_idx)
                A_dict[node_src_idx] = node_dst_idx_lst

            # with open(dataset_path / "A_dict.json", 'w') as write_A:
            #     json.dump(A_dict, write_A)

            # graph indicator dict: {graph_id: [node_idx, ...]}
            graph_indicator = {}
            with open(dataset_path / f"{file_prefix}_graph_indicator.txt", 'r') as \
                    read_G_indicator:
                lines = read_G_indicator.read().splitlines()
            for node_idx, graph_idx in enumerate(lines):
                node_idx_curr = int(node_idx)
                graph_idx_curr = int(graph_idx)
                node_idx_lst = graph_indicator.get(graph_idx_curr, [])
                node_idx_lst.append(node_idx_curr + 1)
                graph_indicator[graph_idx_curr] = node_idx_lst

            # with open(dataset_path /
            # "graph_indicator.json", 'w') as write_G_indicator:
            #     json.dump(graph_indicator, write_G_indicator)

            # features
            features = []
            if dataset_name in ["colors", "triangles"]:
                with open(dataset_path / f"{file_prefix}_node_attributes.txt", 'r') as \
                        read_node_attributes:
                    lines = read_node_attributes.read().splitlines()
                for line in lines:
                    if dataset_name == "colors":
                        # node features matrix: n_nodes x 3 (one_hot encoding for
                        # R,G,B)
                        features_vec = line.split(", ")
                        features_vec = [int(feature)
                                        for feature in features_vec][1:4]
                    elif dataset_name == "triangles":
                        # node features matrix: n_nodes x 1 (number of
                        # triangles)
                        features_vec = [int(line)]
                        features_vec = np.array(features_vec)
                        # num_class = num_class_dict(dataset_name)
                        # one_hot_feature = np.eye(num_class)[features_vec]
                        # features_vec = one_hot_feature
                    features.append(features_vec)
            elif dataset_name == "MUTAG":
                with open(dataset_path / f"{file_prefix}_node_labels.txt", 'r') as \
                    read_node_attributes:
                    lines = read_node_attributes.read().splitlines()
                nodes_label = [int(node_label) for node_label in lines]
                for graph_idx, node_idx_lst in graph_indicator.items():
                    feature_vec_raw = np.array(nodes_label)[np.array(node_idx_lst)-1].reshape(-1)
                    features_vec = np.eye(num_class_dict(dataset_name))[feature_vec_raw]
                    # print(features_vec)
                    # print()
                    # per graph
                    features.append(features_vec)
            
            
            # features = torch.Tensor(features)
            # print(features.shape)

            # labels
            if dataset_name in ["triangles", "MUTAG"]:
                # using graph labels / node labels
                with open(dataset_path /
                          f"{file_prefix}_graph_labels.txt", 'r') as \
                        read_graph_labels:
                    lines = read_graph_labels.read().splitlines()
                if dataset_name == "triangles":
                    labels = [int(line) for line in lines]
                elif dataset_name == "MUTAG":
                    labels = [int(int(line)==1) for line in lines]

            # formatting as Pytorch Geometric data
            # list of Data objects
            graph_data_list = []
            for graph_idx, node_idx_lst in tqdm(graph_indicator.items()):
                graph_x = []
                graph_edge_idx_src = []
                graph_edge_idx_dst = []
                if dataset_name == "colors":
                    count_green = 0
                for node_idx in node_idx_lst:
                    if dataset_name != "MUTAG":
                        node_feature_vec = features[node_idx - 1]
                        graph_x.append(features[node_idx - 1])
                    if dataset_name == "colors":
                        if node_feature_vec[1] == 1:
                            count_green += 1
                    if node_idx in A_dict:
                        # has edges pointing out
                        for node_idx_dst in A_dict[node_idx]:
                            graph_edge_idx_src.append(node_idx - node_idx_lst[0]) 
                            graph_edge_idx_dst.append(node_idx_dst - node_idx_lst[0]) 
                if dataset_name == "MUTAG":
                    graph_x = torch.tensor(features[graph_idx-1])
                else:
                    graph_x = torch.tensor(graph_x, dtype=torch.float)
                graph_edge_idx = torch.tensor(
                    [graph_edge_idx_src, graph_edge_idx_dst], dtype=torch.long)

                graph_data = Data(
                    x=graph_x,
                    edge_index=graph_edge_idx,
                    y=count_green if
                    dataset_name == "colors" else labels[graph_idx - 1])
                graph_data_list.append(graph_data)

            with open(dataset_path / "preprocessed_data.pkl", 'wb') as f:
                pickle.dump(graph_data_list, f)

        return graph_data_list


def get_loader(
        dataset: str,
        mode: int = 0,
        batch_size: int = 1,
        shuffle: bool = True):
    """Mode: 0 for train, 1 for val, 2 for test, -1 for all"""

    if dataset == "geometric_shapes":
        # GeometricShapes
        # Dataset hold mesh faces instead of edge indices
        # the authors does not break it into train/test, does not support @mode
        dataset = GeometricShapes(root='geometric_shapes')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
    else:
        # Colors
        dataset_path = Path("datasets") / dataset
        if not os.path.exists(str(dataset_path / "preprocessed_data.pkl")):
            graph_data_list = preprocessed_data(
                dataset, dataset_prefix_dict(dataset))
        else:
            print(f"Loading preprocessed data for dataset {dataset}")

            with open(dataset_path / "preprocessed_data.pkl", 'rb') as f:
                graph_data_list = pickle.load(f)
                # print()

        num_data = len(graph_data_list)
        num_train = int(0.8 * num_data)
        num_val = int(0.1 * num_data)

        if shuffle:
            random.shuffle(graph_data_list)
        
        if mode == -1:
            return DataLoader(
                graph_data_list,
                batch_size=batch_size,
                shuffle=shuffle)
        elif mode == 0:
            return DataLoader(
                graph_data_list[:num_train],
                batch_size=batch_size,
                shuffle=shuffle)
        elif mode == 1:
            return DataLoader(
                graph_data_list[num_train:num_train + num_val],
                batch_size=batch_size,
                shuffle=shuffle)
        else:
            return DataLoader(
                graph_data_list[num_train + num_val:],
                batch_size=batch_size,
                shuffle=shuffle)


if __name__ == "__main__":
    # _ = get_loader("geometric_shapes", train=True)
    # print()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "geometric_shapes",
            "colors",
            "triangles",
            "acyclic",
            "MUTAG",
            "all"],
        required=True)

    args = parser.parse_args()

    if args.dataset == "all":
        preprocessing_for_all()
    else:
        loader = get_loader(args.dataset)
    # print()
