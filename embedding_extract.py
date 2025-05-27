# extract_graph_embeddings.py

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
from model import GCNGraphEmbedder  

# -------- Load and convert graphs --------
def load_graph_as_pyg(file_path, label, start_id=0):
    G = nx.read_edgelist(file_path, nodetype=int)
    mapping = {node: i for i, node in enumerate(G.nodes(), start=start_id)}
    G = nx.relabel_nodes(G, mapping)
    data = from_networkx(G)
    num_nodes = G.number_of_nodes()
    data.x = torch.eye(num_nodes)
    data.y = torch.tensor([label], dtype=torch.long)
    return data, start_id + num_nodes

# -------- Main: Extract embeddings --------
def extract_graph_embeddings():
    nid = 0
    data1, nid = load_graph_as_pyg("datasets/collaboration.txt", label=0, start_id=nid)
    data2, nid = load_graph_as_pyg("datasets/facebook_combined.txt", label=1, start_id=nid)
    data3, nid = load_graph_as_pyg("datasets/enron.txt", label=2, start_id=nid)

    dataset = [data1, data2, data3]
    loader = DataLoader(dataset, batch_size=3)

    in_channels = max([d.x.size(1) for d in dataset])
    model = GCNGraphEmbedder(in_channels=in_channels)
    model.eval()

    # Pad to same feature dim
    for data in dataset:
        if data.x.size(1) < in_channels:
            pad = torch.zeros((data.x.size(0), in_channels - data.x.size(1)))
            data.x = torch.cat([data.x, pad], dim=1)

    with torch.no_grad():
        for batch in loader:
            emb = model(batch.x, batch.edge_index, batch.batch)
            torch.save(emb, "graph_embeddings.pt")
            print("Saved to graph_embeddings.pt, shape:", emb.shape)
            print(emb)

if __name__ == "__main__":
    extract_graph_embeddings()