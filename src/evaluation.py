import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import time
import copy
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation and training pipeline
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 1e-4
    ):
        """
        Initialize the model evaluator
        
        Args:
            model: PyTorch model to evaluate
            device: Device to use for training/evaluation
            early_stopping_patience: Patience for early stopping
            early_stopping_delta: Minimum change for early stopping
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Best model state
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
        # Evaluation results
        self.evaluation_results = {}
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        optimizer_name: str = 'AdamW',
        scheduler_name: str = 'StepLR',
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model with comprehensive monitoring
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            optimizer_name: Optimizer type
            scheduler_name: Learning rate scheduler
            weight_decay: Weight decay for regularization
            class_weights: Weights for class imbalance
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Prepare data loaders
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
        
        # Initialize optimizer
        optimizer = self._get_optimizer(optimizer_name, learning_rate, weight_decay)
        
        # Initialize scheduler
        scheduler = self._get_scheduler(scheduler_name, optimizer)
        
        # Initialize loss function
        criterion = self._get_loss_function(class_weights)
        
        # Early stopping variables
        patience_counter = 0
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['epoch_times'].append(epoch_time)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Early stopping check
            if val_loss < best_val_loss - self.early_stopping_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_val_loss = val_loss
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model state")
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                class_logits, risk_scores = self.model(features, return_attention=False)
            else:
                outputs = self.model(features)
                if isinstance(outputs, tuple):
                    class_logits, risk_scores = outputs
                else:
                    class_logits = outputs
                    risk_scores = None
            
            # Calculate loss
            loss = criterion(class_logits, labels)
            
            # Add risk regression loss if available
            if risk_scores is not None:
                # Convert labels to risk scores (0-1 scale)
                risk_targets = labels.float() / 3.0  # Normalize to [0, 1]
                risk_loss = nn.MSELoss()(risk_scores.squeeze(), risk_targets)
                loss += 0.1 * risk_loss  # Weight the risk loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(class_logits.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                    class_logits, risk_scores = self.model(features, return_attention=False)
                else:
                    outputs = self.model(features)
                    if isinstance(outputs, tuple):
                        class_logits, risk_scores = outputs
                    else:
                        class_logits = outputs
                        risk_scores = None
                
                # Calculate loss
                loss = criterion(class_logits, labels)
                
                if risk_scores is not None:
                    risk_targets = labels.float() / 3.0
                    risk_loss = nn.MSELoss()(risk_scores.squeeze(), risk_targets)
                    loss += 0.1 * risk_loss
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(class_logits.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, test_data: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            test_data: Test dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Starting comprehensive evaluation...")
        
        # Create test loader
        test_loader = self._create_dataloader(test_data, batch_size, shuffle=False)
        
        # Get predictions
        all_predictions, all_labels, all_probabilities, all_risk_scores = self._get_predictions(test_loader)
        
        # Calculate metrics
        results = {}
        
        # Basic classification metrics
        results['accuracy'] = accuracy_score(all_labels, all_predictions)
        results['precision_macro'] = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        results['recall_macro'] = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        results['f1_macro'] = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # Per-class metrics
        results['precision_per_class'] = precision_score(all_labels, all_predictions, average=None, zero_division=0)
        results['recall_per_class'] = recall_score(all_labels, all_predictions, average=None, zero_division=0)
        results['f1_per_class'] = f1_score(all_labels, all_predictions, average=None, zero_division=0)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        class_names = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        results['classification_report'] = classification_report(
            all_labels, all_predictions, 
            target_names=class_names,
            zero_division=0
        )
        
        # ROC AUC (for multiclass)
        if all_probabilities is not None and all_probabilities.shape[1] > 2:
            try:
                results['roc_auc_ovr'] = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
                results['roc_auc_ovo'] = roc_auc_score(all_labels, all_probabilities, multi_class='ovo', average='macro')
            except ValueError:
                results['roc_auc_ovr'] = 0.0
                results['roc_auc_ovo'] = 0.0
        
        # Risk score evaluation (if available)
        if all_risk_scores is not None:
            # Convert labels to normalized risk scores
            risk_targets = np.array(all_labels) / 3.0
            results['risk_mse'] = np.mean((all_risk_scores - risk_targets) ** 2)
            results['risk_mae'] = np.mean(np.abs(all_risk_scores - risk_targets))
            
            # Risk score correlation
            results['risk_correlation'] = np.corrcoef(all_risk_scores, risk_targets)[0, 1]
        
        # Model confidence analysis
        if all_probabilities is not None:
            max_probs = np.max(all_probabilities, axis=1)
            results['mean_confidence'] = np.mean(max_probs)
            results['confidence_std'] = np.std(max_probs)
            
            # Confidence vs accuracy
            correct_mask = (all_predictions == all_labels)
            results['confidence_when_correct'] = np.mean(max_probs[correct_mask])
            results['confidence_when_wrong'] = np.mean(max_probs[~correct_mask]) if np.sum(~correct_mask) > 0 else 0.0
        
        self.evaluation_results = results
        
        print(f"Evaluation completed. Overall accuracy: {results['accuracy']:.4f}")
        
        return results
    
    def cross_validate(
        self,
        data: List[Dict[str, Any]],
        n_splits: int = 5,
        batch_size: int = 32,
        **train_kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation
        
        Args:
            data: Complete dataset
            n_splits: Number of CV folds
            batch_size: Batch size
            **train_kwargs: Training parameters
            
        Returns:
            Cross-validation results
        """
        print(f"Starting {n_splits}-fold cross-validation...")
        
        # Prepare data
        features, labels = self._prepare_cv_data(data)
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'val_loss': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            print(f"Training fold {fold + 1}/{n_splits}")
            
            # Split data
            train_fold_data = [data[i] for i in train_idx]
            val_fold_data = [data[i] for i in val_idx]
            
            # Reset model
            self._reset_model()
            
            # Train on fold
            history = self.train(train_fold_data, val_fold_data, **train_kwargs)
            
            # Evaluate on fold
            fold_results = self.evaluate(val_fold_data, batch_size)
            
            # Store results
            cv_results['accuracy'].append(fold_results['accuracy'])
            cv_results['precision'].append(fold_results['precision_macro'])
            cv_results['recall'].append(fold_results['recall_macro'])
            cv_results['f1_score'].append(fold_results['f1_macro'])
            cv_results['val_loss'].append(min(history['val_loss']))
        
        # Print CV summary
        print("\nCross-Validation Results:")
        for metric, values in cv_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        return cv_results
    
    def _get_predictions(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions on test data"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_risk_scores = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                    class_logits, risk_scores = self.model(features, return_attention=False)
                else:
                    outputs = self.model(features)
                    if isinstance(outputs, tuple):
                        class_logits, risk_scores = outputs
                    else:
                        class_logits = outputs
                        risk_scores = None
                
                # Get predictions and probabilities
                probabilities = torch.softmax(class_logits, dim=1)
                _, predictions = torch.max(class_logits, 1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if risk_scores is not None:
                    all_risk_scores.extend(risk_scores.cpu().numpy().flatten())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities) if all_probabilities else None
        all_risk_scores = np.array(all_risk_scores) if all_risk_scores else None
        
        return all_predictions, all_labels, all_probabilities, all_risk_scores
    
    def _create_dataloader(self, data: List[Dict[str, Any]], batch_size: int, shuffle: bool = False) -> DataLoader:
        """Create DataLoader from dataset"""
        from src.data_loader import SensorFusionDataset
        
        dataset = SensorFusionDataset(data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    def _prepare_cv_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for cross-validation"""
        # Extract basic features and labels for stratification
        features = []
        labels = []
        
        for sample in data:
            # Simple feature extraction for stratification
            risk_level = sample['risk_level']
            if hasattr(risk_level, 'value'):
                label = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}.get(risk_level.value, 0)
            else:
                label = 0
            
            labels.append(label)
            features.append([sample['duration_h']])  # Simple feature for stratification
        
        return np.array(features), np.array(labels)
    
    def _reset_model(self):
        """Reset model parameters"""
        def weight_reset(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.LSTM):
                m.reset_parameters()
        
        self.model.apply(weight_reset)
        self.model.to(self.device)
    
    def _get_optimizer(self, optimizer_name: str, learning_rate: float, weight_decay: float) -> optim.Optimizer:
        """Get optimizer instance"""
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def _get_scheduler(self, scheduler_name: str, optimizer: optim.Optimizer):
        """Get learning rate scheduler"""
        if scheduler_name.lower() == 'steplr':
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        elif scheduler_name.lower() == 'cosineannealinglr':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif scheduler_name.lower() == 'reducelronplateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        elif scheduler_name.lower() == 'none':
            return None
        else:
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    def _get_loss_function(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """Get loss function"""
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        return nn.CrossEntropyLoss(weight=class_weights)
    
    def save_model(self, filepath: str, include_optimizer: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results,
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.evaluation_results = checkpoint.get('evaluation_results', {})
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'model_class': self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
            'device': self.device,
            'best_val_loss': self.best_val_loss
        }
        
        if self.evaluation_results:
            summary['test_accuracy'] = self.evaluation_results.get('accuracy', 0.0)
            summary['test_f1_macro'] = self.evaluation_results.get('f1_macro', 0.0)
        
        return summary

class MetricsCalculator:
    """Utility class for calculating various ML metrics"""
    
    @staticmethod
    def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> pd.DataFrame:
        """Calculate per-class metrics"""
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(len(np.unique(y_true)))]
        
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        support = np.bincount(y_true)
        
        metrics_df = pd.DataFrame({
            'Class': class_names[:len(precision)],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support[:len(precision)]
        })
        
        return metrics_df
    
    @staticmethod
    def calculate_confidence_metrics(probabilities: np.ndarray, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate confidence-based metrics"""
        max_probs = np.max(probabilities, axis=1)
        correct_mask = (predictions == labels)
        
        metrics = {
            'mean_confidence': np.mean(max_probs),
            'confidence_std': np.std(max_probs),
            'confidence_when_correct': np.mean(max_probs[correct_mask]),
            'confidence_when_wrong': np.mean(max_probs[~correct_mask]) if np.sum(~correct_mask) > 0 else 0.0,
            'low_confidence_accuracy': np.mean(correct_mask[max_probs < 0.5]) if np.sum(max_probs < 0.5) > 0 else 0.0,
            'high_confidence_accuracy': np.mean(correct_mask[max_probs > 0.9]) if np.sum(max_probs > 0.9) > 0 else 0.0
        }
        
        return metrics

