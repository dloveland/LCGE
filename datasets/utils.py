from torch_geometric.datasets import GeometricShapes
from torch_geometric.loader import DataLoader


def get_loader(
        dataset: str,
        train: bool,
        batch_size: int = 32,
        shuffle: bool = True):
    if dataset == "geometric_shapes":
        # GeometricShapes
        # Dataset hold mesh faces instead of edge indices
        # the authors does not break it into train/test,
        dataset = GeometricShapes(root='geometric_shapes')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
    elif dataset == "colors":
        # Colors
        pass
    elif dataset == "triangles":
        # Triangles
        pass
    else:
        return None
