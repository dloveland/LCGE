import json
import os
import pickle
from pathlib import Path
from platform import node

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import GeometricShapes
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import argparse


def preprocessing_for_all():
    dataset_names = ["colors", "triangles"]
    dataset_prefixes = ["COLORS-3", "TRIANGLES"]
    for dataset_name, dataset_prefix in zip(dataset_names, dataset_prefixes):
        preprocessed_data(dataset_name, dataset_prefix)


def preprocessed_data(dataset_name, file_prefix):
    dataset_path = Path("datasets") / dataset_name
    print(f"Preprocessing dataset {dataset_name}")

    if not os.path.exists(str(dataset_path / "preprocessed_data.pkl")):
        # edges mapping dict: {node_src_idx: [node_dst_idx, ...]}
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

        # node features
        node_features = []
        with open(dataset_path / f"{file_prefix}_node_attributes.txt", 'r') as \
                read_node_attributes:
            lines = read_node_attributes.read().splitlines()
        for line in lines:
            if dataset_name == "colors":
                # node features matrix: n_nodes x 3 (one_hot encoding for R,G,B)
                features_vec = line.split(", ")
                features_vec = [int(feature) for feature in features_vec][1:4]
            elif dataset_name == "triangles":
                # node features matrix: n_nodes x 1 (number of triangles)
                features_vec = [int(line)]
            node_features.append(features_vec)
        # node_features = torch.Tensor(node_features)
        # print(node_features.shape)

        # graph labels
        if dataset_name == "triangles":
            with open(dataset_path /
                      f"{file_prefix}_graph_labels.txt", 'r') as \
                    read_graph_labels:
                lines = read_graph_labels.read().splitlines()
            graph_labels = [int(line) for line in lines]

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
                node_feature_vec = node_features[node_idx - 1]
                if dataset_name == "colors":
                    if node_feature_vec[1] == 1:
                        count_green += 1
                graph_x.append(node_features[node_idx - 1])
                if node_idx in A_dict:
                    # has edges pointing out
                    for node_idx_dst in A_dict[node_idx]:
                        graph_edge_idx_src.append(node_idx)
                        graph_edge_idx_dst.append(node_idx_dst)
            graph_x = torch.tensor(graph_x, dtype=torch.float)
            graph_edge_idx = torch.tensor(
                [graph_edge_idx_src, graph_edge_idx_dst], dtype=torch.long)

            graph_data = Data(
                x=graph_x,
                edge_index=graph_edge_idx,
                y=count_green if
                dataset_name == "colors" else graph_labels[graph_idx - 1])
            graph_data_list.append(graph_data)

        with open(dataset_path / "preprocessed_data.pkl", 'wb') as f:
            pickle.dump(graph_data_list, f)

        return graph_data_list


def get_loader(
        dataset: str,
        mode: int = 0,
        batch_size: int = 32,
        shuffle: bool = True):
    """Mode: 0 for train, 1 for val, 2 for test"""

    if dataset == "geometric_shapes":
        # GeometricShapes
        # Dataset hold mesh faces instead of edge indices
        # the authors does not break it into train/test,
        dataset = GeometricShapes(root='geometric_shapes')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
    else:
        # Colors
        dataset_path = Path("datasets") / dataset
        if not os.path.exists(str(dataset_path / "preprocessed_data.pkl")):
            graph_data_list = preprocessed_data(
                dataset, "COLORS-3" if dataset == "colors" else "TRIANGLES")
        else:
            print(f"Loading preprocessed data for dataset {dataset}")

            with open(dataset_path / "preprocessed_data.pkl", 'rb') as f:
                graph_data_list = pickle.load(f)
                print()

        return DataLoader(
            graph_data_list,
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
            "all"],
        required=True)

    args = parser.parse_args()

    if args.dataset == "all":
        preprocessing_for_all()
    else:
        loader = get_loader(args.dataset)
    print()
