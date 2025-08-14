import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for sensor fusion"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Apply linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SensorFusionLSTM(nn.Module):
    """
    LSTM-based model for multi-sensor fusion and risk assessment
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = True,
        fusion_technique: Optional[object] = None
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.fusion_technique = fusion_technique
        
        # Handle single feature vector input by adding sequence dimension
        self.input_projection = nn.Linear(input_size, input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Risk level regression (for confidence scoring)
        self.risk_regressor = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_attention=False):
        batch_size = x.size(0)
        
        # Handle 2D input (batch_size, features) by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, features)
        
        seq_len, features = x.size(1), x.size(2)
        
        # Apply fusion technique if provided
        if self.fusion_technique is not None:
            x = self.fusion_technique(x)
        
        # Input projection
        x = self.input_projection(x)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification
        class_logits = self.classifier(context)
        risk_score = self.risk_regressor(context)
        
        if return_attention:
            return class_logits, risk_score, attention_weights.squeeze(-1)
        
        return class_logits, risk_score

class SensorFusionTransformer(nn.Module):
    """
    Transformer-based model for multi-sensor fusion and risk assessment
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1,
        fusion_technique: Optional[object] = None
    ):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.fusion_technique = fusion_technique
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Risk regression head
        self.risk_regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Attention weights for interpretability
        self.register_buffer('attention_weights', None)
    
    def forward(self, x, return_attention=False):
        # Handle 2D input (batch_size, features) by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, features)
        
        # Apply fusion technique if provided
        if self.fusion_technique is not None:
            x = self.fusion_technique(x)
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        if return_attention:
            # Custom forward pass to capture attention
            transformer_out = x
            attention_weights = []
            
            for layer in self.transformer.layers:
                transformer_out, attn_weights = layer.self_attn(
                    transformer_out, transformer_out, transformer_out,
                    need_weights=True
                )
                attention_weights.append(attn_weights)
                transformer_out = layer(transformer_out)
            
            # Average attention weights across layers and heads
            avg_attention = torch.stack(attention_weights).mean(dim=(0, 1))
            self.attention_weights = avg_attention
        else:
            transformer_out = self.transformer(x)
        
        # Global pooling
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        
        # Classification and risk prediction
        class_logits = self.classifier(pooled)
        risk_score = self.risk_regressor(pooled)
        
        if return_attention and self.attention_weights is not None:
            return class_logits, risk_score, self.attention_weights
        
        return class_logits, risk_score

class MultiModalFusionNet(nn.Module):
    """
    Multi-modal fusion network with dedicated encoders for different sensor types
    """
    
    def __init__(
        self,
        sensor_types: List[str],
        encoder_dim: int = 128,
        fusion_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.2,
        fusion_technique: Optional[object] = None
    ):
        super().__init__()
        
        self.sensor_types = sensor_types
        self.encoder_dim = encoder_dim
        self.fusion_dim = fusion_dim
        self.fusion_technique = fusion_technique
        
        # Define sensor feature dimensions based on our feature extraction
        sensor_feature_dims = {
            'PIR': 8,
            'Thermal': 8,
            'Radar': 6,
            'Environmental': 6,
            'Audio': 8,
            'Door': 3
        }
        
        # Sensor-specific encoders
        self.encoders = nn.ModuleDict()
        
        for sensor_type in sensor_types:
            input_dim = sensor_feature_dims.get(sensor_type, 8)
            
            self.encoders[sensor_type] = nn.Sequential(
                nn.Linear(input_dim, encoder_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim, encoder_dim),
                nn.ReLU()
            )
        
        # Cross-modal attention
        self.cross_attention = MultiHeadAttention(
            d_model=encoder_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # Fusion layer
        total_encoder_dim = len(sensor_types) * encoder_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_encoder_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Risk regression head
        self.risk_regressor = nn.Sequential(
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, sensor_masks=None, return_attention=False):
        """
        Forward pass with multi-modal sensor data
        
        Args:
            x: Input tensor [batch_size, features] or [batch_size, seq_len, features]
            sensor_masks: Optional masks for missing sensors
            return_attention: Whether to return attention weights
        """
        batch_size = x.size(0)
        
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Split input into sensor-specific features
        sensor_features = self._split_sensor_features(x)
        
        # Encode each sensor type
        encoded_features = []
        sensor_embeddings = []
        
        for sensor_type, features in sensor_features.items():
            if sensor_type in self.encoders:
                # Average over sequence dimension if present
                if features.dim() == 3:
                    features = features.mean(dim=1)
                
                encoded = self.encoders[sensor_type](features)
                encoded_features.append(encoded)
                sensor_embeddings.append(encoded.unsqueeze(1))  # Add sequence dimension
        
        # Stack sensor embeddings for cross-attention
        if sensor_embeddings:
            sensor_stack = torch.cat(sensor_embeddings, dim=1)  # [batch, num_sensors, encoder_dim]
            
            # Apply cross-modal attention
            attended_features, attention_weights = self.cross_attention(
                sensor_stack, sensor_stack, sensor_stack
            )
            
            # Flatten attended features
            attended_flat = attended_features.view(batch_size, -1)
        else:
            attended_flat = torch.zeros(batch_size, len(self.sensor_types) * self.encoder_dim)
            attention_weights = None
        
        # Apply fusion technique if provided
        if self.fusion_technique is not None:
            attended_flat = self.fusion_technique(attended_flat)
        
        # Fusion and classification
        fused = self.fusion_layer(attended_flat)
        class_logits = self.classifier(fused)
        risk_score = self.risk_regressor(fused)
        
        if return_attention and attention_weights is not None:
            return class_logits, risk_score, attention_weights.mean(dim=1)  # Average over heads
        
        return class_logits, risk_score
    
    def _split_sensor_features(self, x):
        """Split input tensor into sensor-specific features"""
        sensor_features = {}
        start_idx = 0
        
        # Feature dimensions based on our extraction logic
        feature_dims = {
            'PIR': 8,
            'Thermal': 8,
            'Radar': 6,
            'Environmental': 6,
            'Audio': 8,
            'Door': 3
        }
        
        for sensor_type in self.sensor_types:
            dim = feature_dims.get(sensor_type, 8)
            end_idx = start_idx + dim
            
            if end_idx <= x.size(-1):
                sensor_features[sensor_type] = x[..., start_idx:end_idx]
            else:
                # Pad if not enough features
                sensor_features[sensor_type] = torch.zeros(x.size(0), x.size(1), dim, device=x.device)
            
            start_idx = end_idx
        
        return sensor_features

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return x + self.dropout(self.layers(x))

class DeepSensorFusionNet(nn.Module):
    """
    Deep residual network for complex sensor fusion scenarios
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        num_classes: int = 4,
        dropout: float = 0.2,
        fusion_technique: Optional[object] = None
    ):
        super().__init__()
        
        self.fusion_technique = fusion_technique
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output heads
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.risk_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_attention=False):
        # Apply fusion technique if provided
        if self.fusion_technique is not None:
            x = self.fusion_technique(x)
        
        # Handle sequence input (take last timestep or pool)
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            x = x.view(-1, features)  # Flatten sequence
            x = self.input_proj(x)
            x = x.view(batch_size, seq_len, -1)
            
            # Attention pooling over time
            attn_weights = self.attention_pool(x)
            attn_weights = F.softmax(attn_weights, dim=1)
            x = torch.sum(x * attn_weights, dim=1)
        else:
            x = self.input_proj(x)
            attn_weights = None
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output predictions
        class_logits = self.classifier(x)
        risk_score = self.risk_regressor(x)
        
        if return_attention and attn_weights is not None:
            return class_logits, risk_score, attn_weights.squeeze(-1)
        
        return class_logits, risk_score

def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific arguments
        
    Returns:
        PyTorch model instance
    """
    model_registry = {
        'lstm': SensorFusionLSTM,
        'transformer': SensorFusionTransformer,
        'multimodal': MultiModalFusionNet,
        'deep': DeepSensorFusionNet
    }
    
    if model_type.lower() not in model_registry:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_registry[model_type.lower()](**kwargs)

# Model configuration presets
MODEL_CONFIGS = {
    'lstm_small': {
        'model_type': 'lstm',
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': True
    },
    'lstm_large': {
        'model_type': 'lstm',
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.3,
        'bidirectional': True
    },
    'transformer_base': {
        'model_type': 'transformer',
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0.1
    },
    'multimodal_fusion': {
        'model_type': 'multimodal',
        'encoder_dim': 128,
        'fusion_dim': 256,
        'dropout': 0.2
    }
}
