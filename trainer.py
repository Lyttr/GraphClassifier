import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Dict
from model import GNNClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class Trainer:
    def __init__(self, 
                 model: GNNClassifier,
                 learning_rate: float = 0.01,
                 batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.device = device
        self.batch_size = batch_size
        
    def train_epoch(self, train_data: List[Data]) -> float:
        """Single epoch training"""
        self.model.train()
        total_loss = 0
        
        # Create batches
        for i in range(0, len(train_data), self.batch_size):
            batch_data = train_data[i:i + self.batch_size]
            batch = Batch.from_data_list(batch_data).to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(batch_data)
            
        return total_loss / len(train_data)
    
    def validate(self, val_data: List[Data]) -> float:
        """Validation step"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), self.batch_size):
                batch_data = val_data[i:i + self.batch_size]
                batch = Batch.from_data_list(batch_data).to(self.device)
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = F.nll_loss(out, batch.y)
                total_loss += loss.item() * len(batch_data)
                
        return total_loss / len(val_data)
    
    def plot_confusion_matrix(self, cm: np.ndarray, output_dir: str, class_names: List[str]):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{output_dir}/confusion_matrix.png')
        plt.close()
    
    def evaluate(self, test_data: List[Data], output_dir: str) -> Tuple[float, np.ndarray, Dict]:
        """Model evaluation with metrics"""
        self.model.eval()
        correct = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for i in range(0, len(test_data), self.batch_size):
                batch_data = test_data[i:i + self.batch_size]
                batch = Batch.from_data_list(batch_data).to(self.device)
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == batch.y).sum())
                predictions.append(pred.cpu().numpy())
                true_labels.append(batch.y.cpu().numpy())
                
        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)
        accuracy = correct / len(test_data)

        class_names = ['Facebook', 'Enron', 'Collaboration']
        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions, 
                                     target_names=class_names,
                                     output_dict=True)
        
    
        class_distribution = {name: int(np.sum(true_labels == i)) for i, name in enumerate(class_names)}
        report['class_distribution'] = class_distribution
 
        metrics = {}
        for class_name in class_names:
            metrics[class_name] = {
                'precision': float(report[class_name]['precision']),
                'recall': float(report[class_name]['recall']),
                'f1-score': float(report[class_name]['f1-score']),
                'support': int(report[class_name]['support'])
            }
        
        # Add overall metrics
        metrics['accuracy'] = float(report['accuracy'])
        metrics['macro avg'] = {
            'precision': float(report['macro avg']['precision']),
            'recall': float(report['macro avg']['recall']),
            'f1-score': float(report['macro avg']['f1-score']),
            'support': int(report['macro avg']['support'])
        }
        metrics['weighted avg'] = {
            'precision': float(report['weighted avg']['precision']),
            'recall': float(report['weighted avg']['recall']),
            'f1-score': float(report['weighted avg']['f1-score']),
            'support': int(report['weighted avg']['support'])
        }
        metrics['class_distribution'] = class_distribution
        
        self.plot_confusion_matrix(cm, output_dir, class_names)
        
        return accuracy, predictions, metrics
    
    def train(self, 
             train_data: List[Data],
             val_data: List[Data],
             test_data: List[Data],
             num_epochs: int = 100,
             print_interval: int = 10,
             patience: int = 10,
             output_dir: str = 'results') -> Tuple[List[float], List[float], float, Dict]:
        """Model training with evaluation"""
        train_losses = []
        val_losses = []
        best_accuracy = 0
        best_metrics = None
        early_stopping = EarlyStopping(patience=patience)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_data)
            val_loss = self.validate(val_data)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            if (epoch + 1) % print_interval == 0:
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                
                accuracy, _, metrics = self.evaluate(test_data, output_dir)
                print(f'Test Accuracy: {accuracy:.4f}')
                print("\nClassification Report:")
                print(pd.DataFrame(metrics).transpose())
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_metrics = metrics
                    torch.save(self.model.state_dict(), f'{output_dir}/best_model.pth')

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        return train_losses, val_losses, best_accuracy, best_metrics 