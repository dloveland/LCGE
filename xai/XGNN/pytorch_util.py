""" Some helper functions for PyTorch, including:
    - normalize_adj/normalize_adj_: systematically normalize adjacency matrix
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress. """

import numpy as np
import sys
import time
import torch


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # print("===Normalizing adjacency matrix symmetrically===")
    adj = adj.numpy()
    N = adj.shape[0]
    adj = adj + np.eye(N)
    D = np.sum(adj, 0)
    D_hat = np.diag(D ** (-0.5))
    out = np.dot(D_hat, adj).dot(D_hat)
    out = torch.from_numpy(out)
    return out, out


def normalize_adj_(adj):
    """Symmetrically normalize adjacency matrix with minor changes to normalize_adj function."""
    adj = adj.numpy()
    D = np.sum(adj, 0)
    D_hat = np.diag(np.power(D,-0.5))
    out = np.dot(D_hat, adj).dot(D_hat)
    out[np.isnan(out)]=0
    out = torch.from_numpy(out)
    return out, out.float()