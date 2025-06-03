import torch
from torch_geometric.data import Data
import networkx as nx
import random
from typing import List, Tuple
import numpy as np

def load_graph(file_path: str) -> nx.Graph:
    """Load and preprocess graph data"""
    G = nx.read_edgelist(file_path, create_using=nx.Graph())
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, node_mapping)
    return G

def sample_subgraph(G: nx.Graph, size: int = 100) -> nx.Graph:
    """Sample a connected subgraph using BFS"""
    nodes = list(G.nodes())
    if len(nodes) <= size:
        return G
    
    start_node = random.choice(nodes)
    subgraph_nodes = set([start_node])
    
    while len(subgraph_nodes) < size:
        neighbors = set()
        for node in subgraph_nodes:
            neighbors.update(G.neighbors(node))
        neighbors = neighbors - subgraph_nodes
        if not neighbors:
            break
        subgraph_nodes.add(random.choice(list(neighbors)))
    
    return G.subgraph(subgraph_nodes)

def prepare_data(graphs: List[nx.Graph], 
                num_samples: int = 500, 
                subgraph_size: int = 100,
                min_subgraph_size: int = 10) -> List[Data]:
    """Prepare graph data for training"""
    data_list = []
    for i, G in enumerate(graphs):
        for _ in range(num_samples):
            subgraph = sample_subgraph(G, subgraph_size)
            if len(subgraph) < min_subgraph_size:
                continue
                
            mapping = {node: idx for idx, node in enumerate(subgraph.nodes())}
            subgraph = nx.relabel_nodes(subgraph, mapping)
            
            node_features = compute_node_features(subgraph)
            edge_index = torch.tensor(list(subgraph.edges()), dtype=torch.long).t().contiguous()
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor([i])
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
    
    return data_list

def compute_node_features(G: nx.Graph) -> np.ndarray:
    """Extract node features: degree, clustering coefficient, and centrality"""
    n_nodes = len(G)
    features = np.zeros((n_nodes, 3))
    
    for i, node in enumerate(G.nodes()):
        features[i, 0] = G.degree(node)
        features[i, 1] = nx.clustering(G, node)
        features[i, 2] = nx.degree_centrality(G)[node]
    
    return features 