import json
import os
import pickle
from pathlib import Path
from platform import node

import numpy as np
import torch
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.datasets import GeometricShapes
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def preprocessed_data(dataset_name, file_prefix):
    dataset_path = Path("datasets") / dataset_name

    if not os.path.exists(str(dataset_path / "preprocessed_data.pkl")):
        print(f"Preprocessing Dataset {dataset_name}")

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

        # with open(dataset_path / "graph_indicator.json", 'w') as write_G_indicator:
        #     json.dump(graph_indicator, write_G_indicator)

        # node features matrix: n_nodes x 3 (one_hot encoding for R,G,B)
        node_features = []
        with open(dataset_path / f"{file_prefix}_node_attributes.txt", 'r') as \
                read_node_attributes:
            lines = read_node_attributes.read().splitlines()
        for line in lines:
            features_vec = line.split(", ")
            features_vec = [int(feature) for feature in features_vec][1:4]
            node_features.append(features_vec)

        # node_features = torch.Tensor(node_features)
        # print(node_features.shape)
        graph_data_list = []
        for graph_idx, node_idx_lst in tqdm(graph_indicator.items()):
            graph_x = []
            graph_edge_idx_src = []
            graph_edge_idx_dst = []
            for node_idx in node_idx_lst:
                graph_x.append(node_features[node_idx - 1])
                if node_idx in A_dict:
                    # has edges pointing out
                    for node_idx_dst in A_dict[node_idx]:
                        graph_edge_idx_src.append(node_idx)
                        graph_edge_idx_dst.append(node_idx_dst)
            graph_x = torch.tensor(graph_x, dtype=torch.float)
            graph_edge_idx = torch.tensor(
                [graph_edge_idx_src, graph_edge_idx_dst], dtype=torch.long)

            graph_data = Data(x=graph_x, edge_index=graph_edge_idx)
            graph_data_list.append(graph_data)
            # print(graph_x)
            # print(graph_edge_idx)
            # print()

        with open(dataset_path / "preprocessed_data.pkl", 'wb') as f:
            pickle.dump(graph_data_list, f)

        return graph_data_list


def get_loader(
        dataset: str,
        train: bool = True,
        batch_size: int = 32,
        shuffle: bool = True):
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
            print(f"Preprocessing dataset {dataset}")
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

# class ColorsDataset(Dataset):
#     def __init__(self, transform=None, pre_transform=None):
#         super().__init__(transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def processed_file_names(self):
#         return ['data.pt']

#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = [...]

#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    get_loader("colors")
