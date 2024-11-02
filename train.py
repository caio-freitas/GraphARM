from torch_geometric.datasets import ZINC
from tqdm import tqdm
import torch
from torch import nn
import math
import wandb
import os

from models import DiffusionOrderingNetwork, DenoisingNetwork
from utils import NodeMasking
from grapharm import GraphARM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# instanciate the dataset
dataset = ZINC(root='./data/ZINC', transform=None, pre_transform=None)

diff_ord_net = DiffusionOrderingNetwork(node_feature_dim=1,
                                        num_node_types=dataset.x.unique().shape[0],
                                        num_edge_types=dataset.edge_attr.unique().shape[0],
                                        num_layers=3,
                                        out_channels=1,
                                        device=device)

masker = NodeMasking(dataset)


denoising_net = DenoisingNetwork(
    node_feature_dim=dataset.num_features,
    edge_feature_dim=dataset.num_edge_features,
    num_node_types=dataset.x.unique().shape[0],
    num_edge_types=dataset.edge_attr.unique().shape[0],
    num_layers=7,
    # hidden_dim=32,
    device=device
)


wandb.init(
        project="GraphARM",
        group=f"v2.3.1",
        name=f"ZINC_GraphARM",
        config={
            "policy": "train",
            "n_epochs": 10000,
            "batch_size": 1,
            "lr": 1e-3,
        },
        # mode='disabled'
    )

torch.autograd.set_detect_anomaly(True)


grapharm = GraphARM(
    dataset=dataset,
    denoising_network=denoising_net,
    diffusion_ordering_network=diff_ord_net,
    device=device
)

batch_size = 5
dataset = dataset[0:5]
try:
    grapharm.load_model()
    print("Loaded model")
except:
    print ("No model to load")
# train loop
for epoch in range(2000):
    print(f"Epoch {epoch}")
    grapharm.train_step(
        train_batch=dataset[2*epoch*batch_size:(2*epoch + 1)*batch_size],
        val_batch=dataset[(2*epoch + 1)*batch_size:batch_size*(2*epoch + 2)],
        M=4
    )
    grapharm.save_model()