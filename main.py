import torch
from model import (
    GNNClassifier, GINClassifier, GATClassifier,
    GraphSAGEClassifier, GraphTransformerClassifier
)
from data_utils import load_graph, prepare_data
from trainer import Trainer
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Classification')
    parser.add_argument('--num-samples', type=int, default=500,
                      help='number of subgraphs to sample from each graph')
    parser.add_argument('--subgraph-size', type=int, default=100,
                      help='size of each subgraph')
    parser.add_argument('--num-epochs', type=int, default=100,
                      help='number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='batch size for training')
    parser.add_argument('--print-interval', type=int, default=10,
                      help='print interval')
    parser.add_argument('--patience', type=int, default=10,
                      help='early stopping patience')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='output directory for results')
    parser.add_argument('--model', type=str, default='gin',
                      choices=['gcn', 'gin', 'gat', 'sage', 'transformer'],
                      help='model architecture to use')
    parser.add_argument('--hidden-dim', type=int, default=64,
                      help='hidden dimension for GNN models')
    parser.add_argument('--num-heads', type=int, default=4,
                      help='number of attention heads for GAT and Transformer models')
    return parser.parse_args()

def save_results(train_losses, val_losses, best_accuracy, best_metrics, args):
    """Save training results and plots"""
    metrics_for_json = {k: v for k, v in best_metrics.items() if k != 'class_distribution'}
    
    results = {
        'model_type': args.model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_accuracy': best_accuracy,
        'best_metrics': metrics_for_json,
        'args': vars(args)
    }
    
    with open(f'{args.output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss ({args.model.upper()})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{args.output_dir}/training_loss.png')
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    graphs = [
        load_graph('datasets/facebook_combined.txt'),
        load_graph('datasets/enron.txt'),
        load_graph('datasets/collaboration.txt')
    ]
    
    data_list = prepare_data(
        graphs,
        num_samples=args.num_samples,
        subgraph_size=args.subgraph_size
    )
    
    # Split data into train, validation and test sets
    train_data, temp_data = train_test_split(data_list, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Initialize model
    if args.model == 'gcn':
        model = GNNClassifier(num_features=3, num_classes=3)
    elif args.model == 'gin':
        model = GINClassifier(num_features=3, num_classes=3, hidden_dim=args.hidden_dim)
    elif args.model == 'gat':
        model = GATClassifier(num_features=3, num_classes=3, 
                            hidden_dim=args.hidden_dim,
                            num_heads=args.num_heads)
    elif args.model == 'sage':
        model = GraphSAGEClassifier(num_features=3, num_classes=3,
                                  hidden_dim=args.hidden_dim)
    else:  # transformer
        model = GraphTransformerClassifier(num_features=3, num_classes=3,
                                         hidden_dim=args.hidden_dim,
                                         num_heads=args.num_heads)
    
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    train_losses, val_losses, best_accuracy, best_metrics = trainer.train(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        num_epochs=args.num_epochs,
        print_interval=args.print_interval,
        patience=args.patience,
        output_dir=args.output_dir
    )
    
    print("\nFinal Results:")
    print(f"Model: {args.model.upper()}")
    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    
    print("\nClass Distribution:")
    for class_name, count in best_metrics['class_distribution'].items():
        print(f"{class_name}: {count} samples")
    
    print("\nDetailed Metrics:")
    print("\nPer-class metrics:")
    for class_name in ['Facebook', 'Enron', 'Collaboration']:
        metrics = best_metrics[class_name]
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    print("\nOverall metrics:")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print("\nMacro average:")
    macro = best_metrics['macro avg']
    print(f"  Precision: {macro['precision']:.4f}")
    print(f"  Recall: {macro['recall']:.4f}")
    print(f"  F1-score: {macro['f1-score']:.4f}")
    print(f"  Support: {macro['support']}")
    
    print("\nWeighted average:")
    weighted = best_metrics['weighted avg']
    print(f"  Precision: {weighted['precision']:.4f}")
    print(f"  Recall: {weighted['recall']:.4f}")
    print(f"  F1-score: {weighted['f1-score']:.4f}")
    print(f"  Support: {weighted['support']}")
    
    save_results(train_losses, val_losses, best_accuracy, best_metrics, args)

if __name__ == '__main__':
    main() 