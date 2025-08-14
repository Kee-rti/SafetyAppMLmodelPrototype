import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

class ModelInterpreter:
    """
    Comprehensive model interpretation and explainability toolkit
    """
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Initialize the model interpreter
        
        Args:
            model: PyTorch model to interpret
            device: Device for computations
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Feature names for interpretation
        self.feature_names = self._get_default_feature_names()
        
        # Class names
        self.class_names = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        
        # Sensor groups for analysis
        self.sensor_groups = {
            'PIR': list(range(0, 8)),
            'Thermal': list(range(8, 16)),
            'Radar': list(range(16, 22)),
            'Environmental': list(range(22, 28)),
            'Audio': list(range(28, 36)),
            'Door': list(range(36, 39))
        }
    
    def shap_analysis(self, data: List[Dict[str, Any]], num_samples: int = 100) -> Dict[str, Any]:
        """
        Perform SHAP analysis for model interpretability
        
        Args:
            data: Dataset for analysis
            num_samples: Number of samples to analyze
            
        Returns:
            SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for this analysis. Install with: pip install shap")
        
        print("Performing SHAP analysis...")
        
        # Prepare data
        features, labels = self._prepare_data_for_analysis(data, num_samples)
        
        # Create SHAP explainer
        explainer = shap.DeepExplainer(self.model, features[:10])  # Use subset as background
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features[:num_samples])
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class output
            shap_results = {
                'shap_values': shap_values,
                'base_values': explainer.expected_value,
                'features': features[:num_samples],
                'labels': labels[:num_samples],
                'feature_names': self.feature_names
            }
        else:
            # Single output
            shap_results = {
                'shap_values': [shap_values],
                'base_values': [explainer.expected_value],
                'features': features[:num_samples],
                'labels': labels[:num_samples],
                'feature_names': self.feature_names
            }
        
        return shap_results
    
    def plot_shap_summary(self, shap_results: Dict[str, Any]) -> go.Figure:
        """
        Create SHAP summary plot
        
        Args:
            shap_results: Results from SHAP analysis
            
        Returns:
            Plotly figure
        """
        shap_values = shap_results['shap_values']
        features = shap_results['features']
        feature_names = shap_results['feature_names']
        
        # Use first class for summary (or average across classes)
        if len(shap_values) > 1:
            # Average SHAP values across classes
            avg_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            avg_shap = np.abs(shap_values[0])
        
        # Calculate feature importance
        feature_importance = np.mean(avg_shap, axis=0)
        
        # Create summary plot
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(feature_importance)],
            'Importance': feature_importance,
            'Sensor_Group': [self._get_sensor_group(i) for i in range(len(feature_importance))]
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df.tail(20),  # Top 20 features
            x='Importance',
            y='Feature',
            color='Sensor_Group',
            title='SHAP Feature Importance Summary',
            labels={'Importance': 'Mean SHAP Value (|importance|)'},
            orientation='h'
        )
        
        fig.update_layout(height=600, showlegend=True)
        
        return fig
    
    def plot_shap_waterfall(self, shap_results: Dict[str, Any], sample_idx: int = 0) -> go.Figure:
        """
        Create SHAP waterfall plot for single prediction
        
        Args:
            shap_results: Results from SHAP analysis
            sample_idx: Index of sample to explain
            
        Returns:
            Plotly figure
        """
        shap_values = shap_results['shap_values'][0]  # Use first class
        base_value = shap_results['base_values'][0] if isinstance(shap_results['base_values'], list) else shap_results['base_values']
        features = shap_results['features']
        feature_names = shap_results['feature_names']
        
        # Get SHAP values for specific sample
        sample_shap = shap_values[sample_idx]
        sample_features = features[sample_idx]
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(sample_shap))[::-1][:15]  # Top 15 features
        
        # Prepare waterfall data
        waterfall_data = []
        cumulative = base_value
        
        for i, idx in enumerate(sorted_indices):
            waterfall_data.append({
                'feature': feature_names[idx],
                'shap_value': sample_shap[idx],
                'feature_value': sample_features[idx],
                'cumulative': cumulative + sample_shap[idx]
            })
            cumulative += sample_shap[idx]
        
        # Create waterfall plot
        fig = go.Figure()
        
        x_pos = list(range(len(waterfall_data)))
        y_values = [d['shap_value'] for d in waterfall_data]
        colors = ['red' if v < 0 else 'green' for v in y_values]
        
        fig.add_trace(go.Bar(
            x=x_pos,
            y=y_values,
            marker_color=colors,
            text=[f"{d['feature']}<br>Value: {d['feature_value']:.3f}<br>SHAP: {d['shap_value']:.3f}" 
                  for d in waterfall_data],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'SHAP Waterfall Plot - Sample {sample_idx}',
            xaxis_title='Features',
            yaxis_title='SHAP Value',
            showlegend=False,
            height=600
        )
        
        return fig
    
    def get_attention_analysis(self, data: List[Dict[str, Any]], num_samples: int = 50) -> np.ndarray:
        """
        Analyze attention weights (for Transformer models)
        
        Args:
            data: Dataset for analysis
            num_samples: Number of samples to analyze
            
        Returns:
            Attention weights array
        """
        if not hasattr(self.model, 'get_attention_weights') and 'Transformer' not in self.model.__class__.__name__:
            print("Model does not support attention analysis")
            return np.zeros((num_samples, 30))  # Return dummy array
        
        print("Analyzing attention weights...")
        
        # Prepare data
        features, _ = self._prepare_data_for_analysis(data, num_samples)
        
        attention_weights = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(features))):
                sample = features[i:i+1]  # Single sample with batch dimension
                
                # Get attention weights
                if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                    _, _, attention = self.model(sample, return_attention=True)
                    if attention is not None:
                        attention_weights.append(attention.cpu().numpy().squeeze())
                    else:
                        attention_weights.append(np.zeros(sample.shape[1]))
                else:
                    attention_weights.append(np.zeros(sample.shape[1]))
        
        return np.array(attention_weights)
    
    def analyze_sensor_importance(self, data: List[Dict[str, Any]], num_samples: int = 100) -> Dict[str, float]:
        """
        Analyze importance of different sensor types
        
        Args:
            data: Dataset for analysis
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary of sensor importance scores
        """
        print("Analyzing sensor importance...")
        
        # Prepare data
        features, labels = self._prepare_data_for_analysis(data, num_samples)
        
        # Calculate baseline performance
        baseline_accuracy = self._calculate_accuracy(features, labels)
        
        # Test importance by masking each sensor group
        sensor_importance = {}
        
        for sensor_name, sensor_indices in self.sensor_groups.items():
            # Create masked features
            masked_features = features.clone()
            
            # Mask sensor features (set to zero)
            if max(sensor_indices) < masked_features.shape[-1]:
                masked_features[:, :, sensor_indices] = 0
            
            # Calculate accuracy with masked sensor
            masked_accuracy = self._calculate_accuracy(masked_features, labels)
            
            # Importance = drop in accuracy when sensor is removed
            importance = baseline_accuracy - masked_accuracy
            sensor_importance[sensor_name] = importance
        
        return sensor_importance
    
    def analyze_risk_scenarios(self, data: List[Dict[str, Any]], num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Analyze model predictions across different risk scenarios
        
        Args:
            data: Dataset for analysis
            num_samples: Number of samples to analyze
            
        Returns:
            List of scenario analysis results
        """
        print("Analyzing risk scenarios...")
        
        # Group data by scenario
        scenario_groups = {}
        for sample in data[:num_samples]:
            scenario = sample['scenario'].value if hasattr(sample['scenario'], 'value') else str(sample['scenario'])
            if scenario not in scenario_groups:
                scenario_groups[scenario] = []
            scenario_groups[scenario].append(sample)
        
        scenario_analysis = []
        
        for scenario, samples in scenario_groups.items():
            if len(samples) < 5:  # Skip scenarios with too few samples
                continue
            
            # Prepare features for this scenario
            features, labels = self._prepare_data_for_analysis(samples, len(samples))
            
            # Get predictions
            with torch.no_grad():
                if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                    class_logits, risk_scores = self.model(features, return_attention=False)
                else:
                    outputs = self.model(features)
                    if isinstance(outputs, tuple):
                        class_logits, risk_scores = outputs
                    else:
                        class_logits = outputs
                        risk_scores = None
                
                probabilities = torch.softmax(class_logits, dim=1)
                predictions = torch.argmax(class_logits, dim=1)
            
            # Calculate metrics
            accuracy = (predictions == labels).float().mean().item()
            confidence = probabilities.max(dim=1)[0].mean().item()
            
            # Risk score analysis
            if risk_scores is not None:
                avg_risk_score = risk_scores.mean().item()
                risk_std = risk_scores.std().item()
            else:
                avg_risk_score = 0.0
                risk_std = 0.0
            
            scenario_analysis.append({
                'scenario': scenario,
                'num_samples': len(samples),
                'accuracy': accuracy,
                'confidence': confidence,
                'avg_risk_score': avg_risk_score,
                'risk_std': risk_std,
                'predicted_classes': predictions.cpu().numpy().tolist(),
                'true_classes': labels.cpu().numpy().tolist()
            })
        
        return scenario_analysis
    
    def create_feature_interaction_plot(self, data: List[Dict[str, Any]], feature_pairs: List[Tuple[int, int]] = None) -> go.Figure:
        """
        Create feature interaction analysis plot
        
        Args:
            data: Dataset for analysis
            feature_pairs: Specific feature pairs to analyze
            
        Returns:
            Plotly figure showing feature interactions
        """
        print("Analyzing feature interactions...")
        
        # Prepare data
        features, labels = self._prepare_data_for_analysis(data, 200)
        
        if feature_pairs is None:
            # Select top features from different sensor groups
            feature_pairs = [
                (2, 10),   # PIR mean vs Thermal mean
                (16, 22),  # Radar range vs Environmental temp
                (28, 36),  # Audio amplitude vs Door state
            ]
        
        # Create subplots
        fig = make_subplots(
            rows=len(feature_pairs), cols=1,
            subplot_titles=[f"{self.feature_names[pair[0]]} vs {self.feature_names[pair[1]]}" 
                           for pair in feature_pairs],
            vertical_spacing=0.1
        )
        
        for i, (feat1_idx, feat2_idx) in enumerate(feature_pairs):
            if feat1_idx < features.shape[-1] and feat2_idx < features.shape[-1]:
                # Extract features (average over time dimension)
                feat1_values = features[:, :, feat1_idx].mean(dim=1).cpu().numpy()
                feat2_values = features[:, :, feat2_idx].mean(dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # Create scatter plot
                for class_idx, class_name in enumerate(self.class_names):
                    mask = labels_np == class_idx
                    if np.sum(mask) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=feat1_values[mask],
                                y=feat2_values[mask],
                                mode='markers',
                                name=f'{class_name}' if i == 0 else '',
                                showlegend=(i == 0),
                                marker=dict(size=6, opacity=0.7),
                                legendgroup=class_name
                            ),
                            row=i+1, col=1
                        )
        
        fig.update_layout(height=300 * len(feature_pairs), title="Feature Interaction Analysis")
        
        return fig
    
    def generate_counterfactual_explanations(
        self, 
        data: List[Dict[str, Any]], 
        target_class: int, 
        num_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual explanations
        
        Args:
            data: Dataset for analysis
            target_class: Target class for counterfactuals
            num_samples: Number of counterfactuals to generate
            
        Returns:
            List of counterfactual explanations
        """
        print(f"Generating counterfactual explanations for class {target_class}...")
        
        # Prepare data
        features, labels = self._prepare_data_for_analysis(data, 100)
        
        # Find samples not in target class
        non_target_mask = labels != target_class
        non_target_features = features[non_target_mask]
        non_target_labels = labels[non_target_mask]
        
        if len(non_target_features) == 0:
            return []
        
        counterfactuals = []
        
        for i in range(min(num_samples, len(non_target_features))):
            original_sample = non_target_features[i:i+1]
            original_label = non_target_labels[i]
            
            # Generate counterfactual by optimization
            counterfactual = self._optimize_counterfactual(
                original_sample, target_class, max_iterations=100
            )
            
            if counterfactual is not None:
                # Calculate changes
                changes = self._calculate_feature_changes(original_sample, counterfactual)
                
                counterfactuals.append({
                    'original_class': original_label.item(),
                    'target_class': target_class,
                    'original_features': original_sample.cpu().numpy(),
                    'counterfactual_features': counterfactual.cpu().numpy(),
                    'feature_changes': changes,
                    'success': True
                })
            else:
                counterfactuals.append({
                    'original_class': original_label.item(),
                    'target_class': target_class,
                    'success': False
                })
        
        return counterfactuals
    
    def _prepare_data_for_analysis(self, data: List[Dict[str, Any]], num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for analysis"""
        from src.data_loader import SensorFusionDataset
        
        # Limit data size
        limited_data = data[:num_samples]
        
        # Create dataset
        dataset = SensorFusionDataset(limited_data)
        
        # Extract features and labels
        features = torch.stack([dataset[i][0] for i in range(len(dataset))])
        labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        
        return features.to(self.device), labels.to(self.device)
    
    def _calculate_accuracy(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate model accuracy on given features and labels"""
        with torch.no_grad():
            if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                class_logits, _ = self.model(features, return_attention=False)
            else:
                outputs = self.model(features)
                if isinstance(outputs, tuple):
                    class_logits, _ = outputs
                else:
                    class_logits = outputs
            
            predictions = torch.argmax(class_logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            
        return accuracy
    
    def _get_sensor_group(self, feature_index: int) -> str:
        """Get sensor group for a feature index"""
        for sensor_name, indices in self.sensor_groups.items():
            if feature_index in indices:
                return sensor_name
        return 'Unknown'
    
    def _get_default_feature_names(self) -> List[str]:
        """Get default feature names"""
        return [
            # PIR features (0-7)
            'pir_mean', 'pir_std', 'pir_max', 'pir_min', 'pir_median',
            'pir_sum', 'pir_above_avg_count', 'pir_variance',
            
            # Thermal features (8-15)
            'thermal_mean', 'thermal_std', 'thermal_max', 'thermal_min',
            'thermal_p95', 'thermal_anomaly_count', 'thermal_variance', 'thermal_gradient',
            
            # Radar features (16-21)
            'radar_range_mean', 'radar_range_std',
            'radar_velocity_mean', 'radar_velocity_std',
            'radar_strength_mean', 'radar_strength_max',
            
            # Environmental features (22-27)
            'env_temp_mean', 'env_temp_std',
            'env_humidity_mean', 'env_humidity_std',
            'env_co2_mean', 'env_co2_std',
            
            # Audio features (28-35)
            'audio_amplitude_mean', 'audio_amplitude_max', 'audio_amplitude_std',
            'audio_freq_mean', 'audio_freq_max',
            'audio_centroid_mean', 'audio_centroid_std', 'audio_active_count',
            
            # Door features (36-38)
            'door_open_pct', 'door_max_duration', 'door_field_strength'
        ]
    
    def _optimize_counterfactual(
        self, 
        original_sample: torch.Tensor, 
        target_class: int, 
        max_iterations: int = 100
    ) -> Optional[torch.Tensor]:
        """Optimize counterfactual example using gradient descent"""
        # Clone original sample
        counterfactual = original_sample.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.Adam([counterfactual], lr=0.01)
        
        target_tensor = torch.tensor([target_class]).to(self.device)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                class_logits, _ = self.model(counterfactual, return_attention=False)
            else:
                outputs = self.model(counterfactual)
                if isinstance(outputs, tuple):
                    class_logits, _ = outputs
                else:
                    class_logits = outputs
            
            # Loss: encourage target class + minimize changes
            class_loss = nn.CrossEntropyLoss()(class_logits, target_tensor)
            distance_loss = torch.norm(counterfactual - original_sample)
            
            total_loss = class_loss + 0.1 * distance_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Check if target class is predicted
            predicted_class = torch.argmax(class_logits, dim=1)
            if predicted_class == target_class:
                return counterfactual.detach()
        
        return None  # Failed to find counterfactual
    
    def _calculate_feature_changes(
        self, 
        original: torch.Tensor, 
        counterfactual: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Calculate changes between original and counterfactual"""
        diff = (counterfactual - original).cpu().numpy().flatten()
        
        changes = []
        for i, change in enumerate(diff):
            if abs(change) > 1e-3:  # Only significant changes
                changes.append({
                    'feature_index': i,
                    'feature_name': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                    'original_value': original.cpu().numpy().flatten()[i],
                    'counterfactual_value': counterfactual.cpu().numpy().flatten()[i],
                    'change': change
                })
        
        # Sort by absolute change
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return changes

class GradientBasedExplainer:
    """Gradient-based explanation methods"""
    
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def integrated_gradients(
        self, 
        inputs: torch.Tensor, 
        target_class: int, 
        steps: int = 50
    ) -> torch.Tensor:
        """
        Calculate Integrated Gradients
        
        Args:
            inputs: Input tensor
            target_class: Target class for explanation
            steps: Number of integration steps
            
        Returns:
            Integrated gradients
        """
        # Baseline (zeros)
        baseline = torch.zeros_like(inputs)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        integrated_grads = torch.zeros_like(inputs)
        
        for alpha in alphas:
            # Interpolated input
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                class_logits, _ = self.model(interpolated, return_attention=False)
            else:
                outputs = self.model(interpolated)
                if isinstance(outputs, tuple):
                    class_logits, _ = outputs
                else:
                    class_logits = outputs
            
            # Gradient w.r.t. target class
            target_score = class_logits[:, target_class].sum()
            gradients = torch.autograd.grad(target_score, interpolated)[0]
            
            integrated_grads += gradients
        
        # Average gradients and multiply by input difference
        integrated_grads /= steps
        integrated_grads *= (inputs - baseline)
        
        return integrated_grads

