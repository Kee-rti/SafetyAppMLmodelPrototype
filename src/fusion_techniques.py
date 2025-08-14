import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class FusionTechnique(nn.Module):
    """Base class for sensor fusion techniques"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_fusion_weights(self):
        """Return current fusion weights for interpretability"""
        return None

class EarlyFusion(FusionTechnique):
    """
    Early fusion: Concatenate raw sensor data before processing
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        
    def forward(self, x):
        """
        Apply early fusion by concatenating sensor features
        
        Args:
            x: Input tensor [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
            
        Returns:
            Fused tensor with same dimensions
        """
        if self.normalize:
            # Normalize each feature dimension
            if x.dim() == 3:
                # For sequence data
                mean = x.mean(dim=(0, 1), keepdim=True)
                std = x.std(dim=(0, 1), keepdim=True) + 1e-8
            else:
                # For batch data
                mean = x.mean(dim=0, keepdim=True)
                std = x.std(dim=0, keepdim=True) + 1e-8
            
            x = (x - mean) / std
        
        return x

class FeatureLevelFusion(FusionTechnique):
    """
    Feature-level fusion: Extract features from each sensor, then fuse
    """
    
    def __init__(
        self, 
        input_dim: int,
        sensor_dims: Dict[str, int],
        fusion_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.sensor_dims = sensor_dims
        self.fusion_dim = fusion_dim
        
        # Feature extractors for each sensor type
        self.feature_extractors = nn.ModuleDict()
        
        start_idx = 0
        for sensor_name, sensor_dim in sensor_dims.items():
            self.feature_extractors[sensor_name] = nn.Sequential(
                nn.Linear(sensor_dim, fusion_dim // len(sensor_dims)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // len(sensor_dims), fusion_dim // len(sensor_dims)),
                nn.ReLU()
            )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(fusion_dim, input_dim)
        
    def forward(self, x):
        """
        Apply feature-level fusion
        
        Args:
            x: Input tensor [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
            
        Returns:
            Fused tensor
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, feature_dim = x.shape
            x = x.view(-1, feature_dim)  # Flatten for processing
        
        # Split input into sensor-specific features
        sensor_features = self._split_features(x)
        
        # Extract features from each sensor
        extracted_features = []
        for sensor_name, features in sensor_features.items():
            if sensor_name in self.feature_extractors:
                extracted = self.feature_extractors[sensor_name](features)
                extracted_features.append(extracted)
        
        # Concatenate extracted features
        if extracted_features:
            fused_features = torch.cat(extracted_features, dim=-1)
        else:
            fused_features = torch.zeros(x.size(0), self.fusion_dim, device=x.device)
        
        # Apply fusion layer
        fused = self.fusion_layer(fused_features)
        
        # Project back to original dimension
        output = self.output_proj(fused)
        
        # Reshape to original shape
        if len(original_shape) == 3:
            output = output.view(original_shape)
        
        return output
    
    def _split_features(self, x):
        """Split input tensor into sensor-specific features"""
        sensor_features = {}
        start_idx = 0
        
        for sensor_name, sensor_dim in self.sensor_dims.items():
            end_idx = start_idx + sensor_dim
            if end_idx <= x.size(-1):
                sensor_features[sensor_name] = x[..., start_idx:end_idx]
            else:
                # Handle case where not enough features
                sensor_features[sensor_name] = torch.zeros(x.size(0), sensor_dim, device=x.device)
            start_idx = end_idx
        
        return sensor_features

class LateFusion(FusionTechnique):
    """
    Late fusion: Process each sensor independently, then fuse predictions
    """
    
    def __init__(
        self,
        input_dim: int,
        sensor_dims: Dict[str, int],
        hidden_dim: int = 64,
        fusion_method: str = 'weighted_average',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.sensor_dims = sensor_dims
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        # Independent processors for each sensor
        self.sensor_processors = nn.ModuleDict()
        
        for sensor_name, sensor_dim in sensor_dims.items():
            self.sensor_processors[sensor_name] = nn.Sequential(
                nn.Linear(sensor_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)  # Output same dim as input
            )
        
        # Fusion weights (learnable)
        if fusion_method == 'weighted_average':
            self.fusion_weights = nn.Parameter(torch.ones(len(sensor_dims)))
        elif fusion_method == 'attention':
            self.attention_net = nn.Sequential(
                nn.Linear(input_dim * len(sensor_dims), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, len(sensor_dims)),
                nn.Softmax(dim=-1)
            )
        
        self.current_weights = None
        
    def forward(self, x):
        """
        Apply late fusion
        
        Args:
            x: Input tensor [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
            
        Returns:
            Fused tensor
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, feature_dim = x.shape
            x = x.view(-1, feature_dim)
        
        # Split and process each sensor independently
        sensor_features = self._split_features(x)
        processed_outputs = []
        
        for sensor_name, features in sensor_features.items():
            if sensor_name in self.sensor_processors:
                processed = self.sensor_processors[sensor_name](features)
                processed_outputs.append(processed)
        
        if not processed_outputs:
            return x.view(original_shape) if len(original_shape) == 3 else x
        
        # Stack outputs for fusion
        stacked_outputs = torch.stack(processed_outputs, dim=-1)  # [batch, features, num_sensors]
        
        # Apply fusion
        if self.fusion_method == 'average':
            fused = stacked_outputs.mean(dim=-1)
            self.current_weights = torch.ones(len(processed_outputs)) / len(processed_outputs)
            
        elif self.fusion_method == 'weighted_average':
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = torch.sum(stacked_outputs * weights, dim=-1)
            self.current_weights = weights.detach().cpu()
            
        elif self.fusion_method == 'attention':
            # Concatenate for attention computation
            concat_features = torch.cat(processed_outputs, dim=-1)
            weights = self.attention_net(concat_features)  # [batch, num_sensors]
            
            # Apply attention weights
            weights = weights.unsqueeze(1)  # [batch, 1, num_sensors]
            fused = torch.sum(stacked_outputs * weights, dim=-1)
            self.current_weights = weights.mean(dim=0).squeeze().detach().cpu()
            
        elif self.fusion_method == 'max':
            fused, _ = stacked_outputs.max(dim=-1)
            
        else:  # Default to average
            fused = stacked_outputs.mean(dim=-1)
        
        # Reshape to original shape
        if len(original_shape) == 3:
            fused = fused.view(original_shape)
        
        return fused
    
    def _split_features(self, x):
        """Split input tensor into sensor-specific features"""
        sensor_features = {}
        start_idx = 0
        
        for sensor_name, sensor_dim in self.sensor_dims.items():
            end_idx = start_idx + sensor_dim
            if end_idx <= x.size(-1):
                sensor_features[sensor_name] = x[..., start_idx:end_idx]
            else:
                sensor_features[sensor_name] = torch.zeros(x.size(0), sensor_dim, device=x.device)
            start_idx = end_idx
        
        return sensor_features
    
    def get_fusion_weights(self):
        """Return current fusion weights for interpretability"""
        return self.current_weights

class DynamicGatedFusion(FusionTechnique):
    """
    Dynamic gated fusion: Adaptive sensor weighting based on input characteristics
    """
    
    def __init__(
        self,
        input_dim: int,
        sensor_dims: Dict[str, int],
        gate_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.sensor_dims = sensor_dims
        self.gate_dim = gate_dim
        self.temperature = temperature
        self.num_sensors = len(sensor_dims)
        
        # Sensor-specific feature extractors
        self.sensor_encoders = nn.ModuleDict()
        
        for sensor_name, sensor_dim in sensor_dims.items():
            self.sensor_encoders[sensor_name] = nn.Sequential(
                nn.Linear(sensor_dim, gate_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gate_dim, gate_dim)
            )
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, gate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_dim, gate_dim // 2),
            nn.ReLU(),
            nn.Linear(gate_dim // 2, self.num_sensors)
        )
        
        # Quality assessment network
        self.quality_net = nn.ModuleDict()
        for sensor_name in sensor_dims.keys():
            self.quality_net[sensor_name] = nn.Sequential(
                nn.Linear(gate_dim, gate_dim // 2),
                nn.ReLU(),
                nn.Linear(gate_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(gate_dim * self.num_sensors, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.current_gates = None
        self.current_quality = None
        
    def forward(self, x):
        """
        Apply dynamic gated fusion
        
        Args:
            x: Input tensor [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
            
        Returns:
            Fused tensor with adaptive sensor weighting
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, feature_dim = x.shape
            x = x.view(-1, feature_dim)
        
        # Compute input-dependent gates
        gates = self.gate_network(x)  # [batch, num_sensors]
        gates = F.softmax(gates / self.temperature, dim=-1)
        
        # Split and encode sensor features
        sensor_features = self._split_features(x)
        encoded_sensors = []
        quality_scores = []
        
        sensor_names = list(self.sensor_dims.keys())
        
        for i, (sensor_name, features) in enumerate(sensor_features.items()):
            if sensor_name in self.sensor_encoders:
                # Encode sensor features
                encoded = self.sensor_encoders[sensor_name](features)
                encoded_sensors.append(encoded)
                
                # Compute quality score
                quality = self.quality_net[sensor_name](encoded)
                quality_scores.append(quality.squeeze(-1))
        
        if not encoded_sensors:
            return x.view(original_shape) if len(original_shape) == 3 else x
        
        # Stack encoded features and quality scores
        stacked_features = torch.stack(encoded_sensors, dim=1)  # [batch, num_sensors, gate_dim]
        quality_tensor = torch.stack(quality_scores, dim=1)     # [batch, num_sensors]
        
        # Combine gates with quality scores
        combined_weights = gates * quality_tensor
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply adaptive weighting
        weighted_features = stacked_features * combined_weights.unsqueeze(-1)
        
        # Concatenate weighted features
        fused_features = weighted_features.view(x.size(0), -1)
        
        # Final fusion
        fused_output = self.fusion_layer(fused_features)
        
        # Store current weights for interpretability
        self.current_gates = combined_weights.detach().cpu()
        self.current_quality = quality_tensor.detach().cpu()
        
        # Reshape to original shape
        if len(original_shape) == 3:
            fused_output = fused_output.view(original_shape)
        
        return fused_output
    
    def _split_features(self, x):
        """Split input tensor into sensor-specific features"""
        sensor_features = {}
        start_idx = 0
        
        for sensor_name, sensor_dim in self.sensor_dims.items():
            end_idx = start_idx + sensor_dim
            if end_idx <= x.size(-1):
                sensor_features[sensor_name] = x[..., start_idx:end_idx]
            else:
                sensor_features[sensor_name] = torch.zeros(x.size(0), sensor_dim, device=x.device)
            start_idx = end_idx
        
        return sensor_features
    
    def get_fusion_weights(self):
        """Return current fusion weights and quality scores"""
        return {
            'gates': self.current_gates,
            'quality': self.current_quality
        }

class HierarchicalFusion(FusionTechnique):
    """
    Hierarchical fusion: Multi-level fusion with different granularities
    """
    
    def __init__(
        self,
        input_dim: int,
        sensor_dims: Dict[str, int],
        hierarchy_levels: List[List[str]],
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.sensor_dims = sensor_dims
        self.hierarchy_levels = hierarchy_levels
        self.hidden_dim = hidden_dim
        
        # First level: sensor-specific processing
        self.level1_processors = nn.ModuleDict()
        for sensor_name, sensor_dim in sensor_dims.items():
            self.level1_processors[sensor_name] = nn.Sequential(
                nn.Linear(sensor_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Higher levels: group fusion
        self.level_fusers = nn.ModuleList()
        
        for level_sensors in hierarchy_levels:
            level_input_dim = len(level_sensors) * hidden_dim
            self.level_fusers.append(nn.Sequential(
                nn.Linear(level_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Final fusion
        final_input_dim = len(hierarchy_levels) * hidden_dim
        self.final_fuser = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        """Apply hierarchical fusion"""
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, feature_dim = x.shape
            x = x.view(-1, feature_dim)
        
        # Level 1: Process individual sensors
        sensor_features = self._split_features(x)
        level1_outputs = {}
        
        for sensor_name, features in sensor_features.items():
            if sensor_name in self.level1_processors:
                level1_outputs[sensor_name] = self.level1_processors[sensor_name](features)
        
        # Higher levels: Fuse sensor groups
        level_outputs = []
        
        for level_idx, level_sensors in enumerate(self.hierarchy_levels):
            level_features = []
            for sensor_name in level_sensors:
                if sensor_name in level1_outputs:
                    level_features.append(level1_outputs[sensor_name])
            
            if level_features:
                concatenated = torch.cat(level_features, dim=-1)
                fused = self.level_fusers[level_idx](concatenated)
                level_outputs.append(fused)
        
        # Final fusion
        if level_outputs:
            final_concat = torch.cat(level_outputs, dim=-1)
            final_output = self.final_fuser(final_concat)
        else:
            final_output = torch.zeros_like(x)
        
        # Reshape to original shape
        if len(original_shape) == 3:
            final_output = final_output.view(original_shape)
        
        return final_output
    
    def _split_features(self, x):
        """Split input tensor into sensor-specific features"""
        sensor_features = {}
        start_idx = 0
        
        for sensor_name, sensor_dim in self.sensor_dims.items():
            end_idx = start_idx + sensor_dim
            if end_idx <= x.size(-1):
                sensor_features[sensor_name] = x[..., start_idx:end_idx]
            else:
                sensor_features[sensor_name] = torch.zeros(x.size(0), sensor_dim, device=x.device)
            start_idx = end_idx
        
        return sensor_features

def get_fusion_technique(
    fusion_type: str,
    input_dim: int,
    sensor_dims: Optional[Dict[str, int]] = None,
    **kwargs
) -> FusionTechnique:
    """
    Factory function to create fusion techniques
    
    Args:
        fusion_type: Type of fusion technique
        input_dim: Input feature dimension
        sensor_dims: Dictionary mapping sensor names to their feature dimensions
        **kwargs: Additional parameters
        
    Returns:
        Fusion technique instance
    """
    # Default sensor dimensions if not provided
    if sensor_dims is None:
        sensor_dims = {
            'PIR': 3,
            'Thermal': 3,
            'Radar': 3,
            'Environmental': 3,
            'Audio': 3,
            'Door': 2
        }
    
    fusion_registry = {
        'early': lambda: EarlyFusion(**kwargs),
        'feature': lambda: FeatureLevelFusion(input_dim, sensor_dims, **kwargs),
        'late': lambda: LateFusion(input_dim, sensor_dims, **kwargs),
        'dynamic': lambda: DynamicGatedFusion(input_dim, sensor_dims, **kwargs),
        'hierarchical': lambda: HierarchicalFusion(input_dim, sensor_dims, **kwargs)
    }
    
    if fusion_type.lower() not in fusion_registry:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion_registry[fusion_type.lower()]()

# Fusion technique configuration presets
FUSION_CONFIGS = {
    'early_normalized': {
        'fusion_type': 'early',
        'normalize': True
    },
    'feature_level_128': {
        'fusion_type': 'feature',
        'fusion_dim': 128,
        'dropout': 0.1
    },
    'late_attention': {
        'fusion_type': 'late',
        'fusion_method': 'attention',
        'hidden_dim': 64
    },
    'dynamic_adaptive': {
        'fusion_type': 'dynamic',
        'gate_dim': 64,
        'temperature': 0.5
    }
}
