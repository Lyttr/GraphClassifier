# model.py

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNGraphEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        graph_emb = global_mean_pool(x, batch)  # [num_graphs, out_channels]
        return graph_emb