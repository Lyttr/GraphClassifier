import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_model_results(results_dir):
    """Load results for all models from the results directory"""
    models = ['gcn', 'gin', 'gat', 'sage', 'transformer']
    results = {}
    
    for model in models:
        model_dir = os.path.join(results_dir, model)
        results_file = os.path.join(model_dir, 'results.json')
        time_file = os.path.join(model_dir, 'execution_time.txt')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results[model] = json.load(f)
        
        if os.path.exists(time_file):
            with open(time_file, 'r') as f:
                time_info = f.read()
                results[model]['execution_time'] = time_info
    
    return results

def plot_metrics_comparison(results, output_dir):
    """Plot comparison of different metrics across models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    models = list(results.keys())
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Extract metric values for each model
        values = []
        for model in models:
            if metric == 'accuracy':
                values.append(results[model]['best_metrics']['accuracy'])
            else:
                values.append(results[model]['best_metrics']['macro avg'][metric])
        
        # Create bar plot
        sns.barplot(x=models, y=values, ax=ax)
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim(0.9, 1.0)  # Adjust y-axis range for better visualization
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Model')
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()

def plot_loss_comparison(results, output_dir):
    """Plot training and validation loss comparison"""
    plt.figure(figsize=(12, 6))
    
    for model in results.keys():
        train_losses = results[model]['train_losses']
        val_losses = results[model]['val_losses']
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, label=f'{model.upper()} Train')
        plt.plot(epochs, val_losses, label=f'{model.upper()} Val', linestyle='--')
    
    plt.title('Training and Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'))
    plt.close()

def create_comparison_table(results, output_dir):
    """Create a comparison table of all metrics"""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    # Create DataFrame for comparison
    comparison_data = []
    for model in models:
        model_data = {
            'Model': model.upper(),
            'Accuracy': results[model]['best_metrics']['accuracy']
        }
        
        # Add macro average metrics
        for metric in metrics[1:]:  # Skip accuracy as it's already added
            model_data[metric.capitalize()] = results[model]['best_metrics']['macro avg'][metric]
        
        # Add execution time
        if 'execution_time' in results[model]:
            time_str = results[model]['execution_time']
            time_value = float(time_str.split('Execution time: ')[1].split(' seconds')[0])
            model_data['Execution Time (s)'] = time_value
        
        comparison_data.append(model_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(comparison_data)
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Create formatted table for display
    print("\nModel Comparison Summary:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

def main():
    # Load results
    results_dir = 'results_20250603_115754'
    results = load_model_results(results_dir)
    
    # Create comparison directory
    comparison_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Generate comparisons
    plot_metrics_comparison(results, comparison_dir)
    plot_loss_comparison(results, comparison_dir)
    create_comparison_table(results, comparison_dir)
    
    print(f"\nComparison results have been saved to: {comparison_dir}")
    print("Files generated:")
    print("- metrics_comparison.png: Bar plots comparing different metrics")
    print("- loss_comparison.png: Line plots comparing training and validation losses")
    print("- model_comparison.csv: Detailed comparison table")

if __name__ == "__main__":
    main() 