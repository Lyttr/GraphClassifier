# Graph Classification with GNNs

This project implements various Graph Neural Network (GNN) models for graph classification tasks. It supports multiple GNN architectures including GCN, GIN, GAT, GraphSAGE, and Graph Transformer.

## Features

- Multiple GNN architectures:
  - GCN (Graph Convolutional Network)
  - GIN (Graph Isomorphism Network)
  - GAT (Graph Attention Network)
  - GraphSAGE
  - Graph Transformer
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics
- Visualization of training process and results
- Support for batch processing
- Model checkpointing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Lyttr/GraphClassifier.git
cd GraphClassifier
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The project expects graph data in the following format:
- `datasets/facebook_combined.txt`: Facebook social network graph
- `datasets/enron.txt`: Enron email network graph
- `datasets/collaboration.txt`: Scientific collaboration network graph

Each graph file should be in edge list format:
```
node1 node2
node2 node3
...
```

## Usage

### Basic Usage

Train a model with default parameters:
```bash
python main.py --model gin --output-dir results
```

### Advanced Usage

Train with custom parameters:
```bash
python main.py \
    --model gat \
    --num-samples 500 \
    --subgraph-size 100 \
    --num-epochs 100 \
    --batch-size 32 \
    --hidden-dim 64 \
    --num-heads 4 \
    --learning-rate 0.01 \
    --patience 10 \
    --output-dir results
```

### Command Line Arguments

- `--model`: Model architecture to use (`gcn`, `gin`, `gat`, `sage`, `transformer`)
- `--num-samples`: Number of subgraphs to sample from each graph
- `--subgraph-size`: Size of each subgraph
- `--num-epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--hidden-dim`: Hidden dimension for GNN models
- `--num-heads`: Number of attention heads (for GAT and Transformer)
- `--learning-rate`: Learning rate
- `--patience`: Early stopping patience
- `--output-dir`: Output directory for results

## Model Architectures

### GCN (Graph Convolutional Network)


### GIN (Graph Isomorphism Network)


### GAT (Graph Attention Network)

### GraphSAGE

### Graph Transformer

## Output

The training process generates:
1. Training and validation loss plots
2. Confusion matrix
3. Detailed classification metrics
4. Model checkpoints
5. Training results in JSON format

## Results

The evaluation metrics include:
- Per-class precision, recall, and F1-score
- Overall accuracy
- Macro and weighted averages
- Class distribution statistics

## Requirements

- Python 3.7+
- PyTorch 1.8+
- PyTorch Geometric
- scikit-learn
- pandas
- matplotlib
- seaborn

