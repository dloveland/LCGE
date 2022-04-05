import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar

from gnn_explain import gnn_explain

def parse_args():
    parser = argparse.ArgumentParser(description="Run GNN explainer")
    parser.add_argument('--checkpoint', nargs='?', default='/shared-datadrive/shared-training/LCGE/xai/checkpoint/original_XGNN_MUTAG_ckpt.pth',
                        help='path to the true checkpoint')
    parser.add_argument('--save_path', nargs='?', default='/shared-datadrive/shared-training/LCGE/xai/checkpoint',
                        help='the folder where you save the best checkpoint')
    parser.add_argument('--init_graph', nargs='?', default='reset',
                        help='reset: start from an empty graph, load: start from an existing graph')    
    return parser.parse_args()


args = parse_args()

####arguments: (max_node, max_step, target_class, max_iters)
explainer = gnn_explain(args, 6, 30,  1, 50) 

explainer.train()





