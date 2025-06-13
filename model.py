import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, global_mean_pool, GINConv, GATConv,
    SAGEConv, TransformerConv, global_max_pool
)

class GNNClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GINClassifier(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super(GINClassifier, self).__init__()
        
        # MLP for GIN layers
        self.mlp1 = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # GIN layers
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # First GIN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GIN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GATClassifier(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_heads=4):
        super(GATClassifier, self).__init__()
        
        # GAT layers with multi-head attention
        self.conv1 = GATConv(num_features, hidden_dim // num_heads, heads=num_heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GraphSAGEClassifier(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super(GraphSAGEClassifier, self).__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # First GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GraphTransformerClassifier(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_heads=4):
        super(GraphTransformerClassifier, self).__init__()
        
        # Transformer layers
        self.conv1 = TransformerConv(
            num_features, hidden_dim // num_heads,
            heads=num_heads, dropout=0.2
        )
        self.conv2 = TransformerConv(
            hidden_dim, hidden_dim // num_heads,
            heads=num_heads, dropout=0.2
        )
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Multi-head pooling
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # First Transformer layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second Transformer layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Global pooling with attention
        x = global_max_pool(x, batch)
        x = self.pool(x)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1) 