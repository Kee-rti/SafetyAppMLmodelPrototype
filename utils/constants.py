from enum import Enum
from typing import Dict, List, Any

class ScenarioType(Enum):
    """Enumeration of safety monitoring scenarios"""
    NORMAL_ACTIVITY = "normal_activity"
    NO_MOVEMENT = "no_movement"
    PROLONGED_SILENCE = "prolonged_silence"
    FALL_DETECTED = "fall_detected"
    MEDICAL_EMERGENCY = "medical_emergency"
    DOOR_LEFT_OPEN = "door_left_open"
    WANDERING_RISK = "wandering_risk"
    ENTRAPMENT_RISK = "entrapment_risk"
    INTRUSION_ALERT = "intrusion_alert"
    ENVIRONMENTAL_HAZARD = "environmental_hazard"

class RiskLevel(Enum):
    """Enumeration of risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Sensor types supported by the system
SENSOR_TYPES = [
    "PIR",
    "Thermal", 
    "Radar",
    "Environmental",
    "Audio",
    "Door"
]

# Risk level to integer mapping
RISK_LEVEL_MAPPING = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3
}

# Integer to risk level mapping
INTEGER_TO_RISK_LEVEL = {v: k for k, v in RISK_LEVEL_MAPPING.items()}

# Scenario to risk level mapping based on TRS specifications
SCENARIO_RISK_MAPPING = {
    ScenarioType.NORMAL_ACTIVITY: RiskLevel.LOW,
    ScenarioType.NO_MOVEMENT: RiskLevel.MEDIUM,
    ScenarioType.PROLONGED_SILENCE: RiskLevel.HIGH,
    ScenarioType.FALL_DETECTED: RiskLevel.CRITICAL,
    ScenarioType.MEDICAL_EMERGENCY: RiskLevel.CRITICAL,
    ScenarioType.DOOR_LEFT_OPEN: RiskLevel.MEDIUM,
    ScenarioType.WANDERING_RISK: RiskLevel.HIGH,
    ScenarioType.ENTRAPMENT_RISK: RiskLevel.HIGH,
    ScenarioType.INTRUSION_ALERT: RiskLevel.CRITICAL,
    ScenarioType.ENVIRONMENTAL_HAZARD: RiskLevel.HIGH
}

# Model configurations for different architectures
MODEL_CONFIGS = {
    'lstm_small': {
        'model_type': 'lstm',
        'input_size': 39,  # Total features from all sensors
        'hidden_size': 64,
        'num_layers': 2,
        'num_classes': 4,
        'dropout': 0.2,
        'bidirectional': True
    },
    'lstm_medium': {
        'model_type': 'lstm',
        'input_size': 39,
        'hidden_size': 128,
        'num_layers': 3,
        'num_classes': 4,
        'dropout': 0.3,
        'bidirectional': True
    },
    'lstm_large': {
        'model_type': 'lstm',
        'input_size': 39,
        'hidden_size': 256,
        'num_layers': 4,
        'num_classes': 4,
        'dropout': 0.3,
        'bidirectional': True
    },
    'transformer_base': {
        'model_type': 'transformer',
        'input_size': 39,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'num_classes': 4,
        'dropout': 0.1
    },
    'transformer_large': {
        'model_type': 'transformer',
        'input_size': 39,
        'd_model': 512,
        'nhead': 16,
        'num_layers': 6,
        'num_classes': 4,
        'dropout': 0.1
    },
    'multimodal_fusion': {
        'model_type': 'multimodal',
        'sensor_types': SENSOR_TYPES,
        'encoder_dim': 128,
        'fusion_dim': 256,
        'num_classes': 4,
        'dropout': 0.2
    },
    'deep_residual': {
        'model_type': 'deep',
        'input_size': 39,
        'hidden_dim': 256,
        'num_blocks': 4,
        'num_classes': 4,
        'dropout': 0.2
    }
}

# Fusion technique configurations
FUSION_CONFIGS = {
    'early_fusion': {
        'fusion_type': 'early',
        'normalize': True
    },
    'feature_level_fusion': {
        'fusion_type': 'feature',
        'input_dim': 39,
        'sensor_dims': {
            'PIR': 8,
            'Thermal': 8,
            'Radar': 6,
            'Environmental': 6,
            'Audio': 8,
            'Door': 3
        },
        'fusion_dim': 128,
        'dropout': 0.1
    },
    'late_fusion': {
        'fusion_type': 'late',
        'input_dim': 39,
        'sensor_dims': {
            'PIR': 8,
            'Thermal': 8,
            'Radar': 6,
            'Environmental': 6,
            'Audio': 8,
            'Door': 3
        },
        'hidden_dim': 64,
        'fusion_method': 'weighted_average',
        'dropout': 0.1
    },
    'dynamic_gated_fusion': {
        'fusion_type': 'dynamic',
        'input_dim': 39,
        'sensor_dims': {
            'PIR': 8,
            'Thermal': 8,
            'Radar': 6,
            'Environmental': 6,
            'Audio': 8,
            'Door': 3
        },
        'gate_dim': 64,
        'dropout': 0.1,
        'temperature': 1.0
    },
    'hierarchical_fusion': {
        'fusion_type': 'hierarchical',
        'input_dim': 39,
        'sensor_dims': {
            'PIR': 8,
            'Thermal': 8,
            'Radar': 6,
            'Environmental': 6,
            'Audio': 8,
            'Door': 3
        },
        'hierarchy_levels': [
            ['PIR', 'Thermal'],  # Motion sensors
            ['Radar', 'Audio'],  # Active sensors
            ['Environmental', 'Door']  # Context sensors
        ],
        'hidden_dim': 64,
        'dropout': 0.1
    }
}

# Training hyperparameter ranges for optimization
HYPERPARAMETER_RANGES = {
    'learning_rate': (1e-5, 1e-2),
    'batch_size': [16, 32, 64, 128],
    'hidden_size': (32, 512),
    'num_layers': (1, 6),
    'dropout': (0.0, 0.5),
    'weight_decay': (1e-6, 1e-2)
}

# Performance thresholds based on TRS requirements
PERFORMANCE_THRESHOLDS = {
    'minimum_accuracy': 0.998,  # 99.8% as per TRS
    'minimum_precision': 0.995,
    'minimum_recall': 0.995,
    'minimum_f1_score': 0.995,
    'maximum_false_positive_rate': 0.002,
    'maximum_inference_time_ms': 50  # For edge deployment
}

# Sensor specifications from TRS documentation
SENSOR_SPECIFICATIONS = {
    'PIR': {
        'model': 'HC-SR501',
        'detection_range_m': (3, 7),
        'detection_angle_deg': 120,
        'operating_voltage': (4.5, 20.0),
        'power_consumption_mw': 0.165,
        'response_time_s': 2.5,
        'feature_count': 8
    },
    'Thermal': {
        'model': 'AMG8833',
        'resolution': (8, 8),
        'field_of_view_deg': (60, 60),
        'temp_range_c': (0, 80),
        'accuracy_c': 2.5,
        'power_consumption_mw': 14.85,
        'refresh_rate_hz': 10,
        'feature_count': 8
    },
    'Radar': {
        'model': 'BGT60TR13C',
        'frequency_range_ghz': (57, 64),
        'detection_range_m': (0.15, 10),
        'range_resolution_cm': 15,
        'velocity_range_mps': (-8.1, 8.1),
        'power_consumption_mw': 280.5,
        'update_rate_hz': 100,
        'feature_count': 6
    },
    'Environmental': {
        'temperature': {
            'model': 'DS18B20+',
            'range_c': (-55, 125),
            'accuracy_c': 0.5,
            'power_consumption_mw': 4.5
        },
        'humidity': {
            'model': 'SHT40-AD1B',
            'range_pct': (0, 100),
            'accuracy_pct': 1.8,
            'power_consumption_mw': 1.0
        },
        'co2': {
            'model': 'SCD40-D-R2',
            'range_ppm': (0, 40000),
            'accuracy': '±(50 ppm + 5% of reading)',
            'power_consumption_mw': 59.4
        },
        'feature_count': 6
    },
    'Audio': {
        'model': 'INMP441',
        'acoustic_overload_point_db': 120,
        'snr_db': 61,
        'sensitivity_dbfs': -26,
        'frequency_response_hz': (60, 15000),
        'power_consumption_mw': 4.62,
        'feature_count': 8
    },
    'Door': {
        'magnetic': {
            'model': 'MC-38',
            'gap_distance_mm': 20,
            'power_consumption_mw': 0.0
        },
        'hall_effect': {
            'model': 'A3144EUA-T',
            'magnetic_threshold_gauss': {'operate': 280, 'release': 200},
            'power_consumption_mw': 15.0
        },
        'feature_count': 3
    }
}

# Feature names for each sensor type
FEATURE_NAMES = {
    'PIR': [
        'pir_mean', 'pir_std', 'pir_max', 'pir_min', 'pir_median',
        'pir_sum', 'pir_above_avg_count', 'pir_variance'
    ],
    'Thermal': [
        'thermal_mean', 'thermal_std', 'thermal_max', 'thermal_min',
        'thermal_p95', 'thermal_anomaly_count', 'thermal_variance', 'thermal_gradient'
    ],
    'Radar': [
        'radar_range_mean', 'radar_range_std',
        'radar_velocity_mean', 'radar_velocity_std',
        'radar_strength_mean', 'radar_strength_max'
    ],
    'Environmental': [
        'env_temp_mean', 'env_temp_std',
        'env_humidity_mean', 'env_humidity_std',
        'env_co2_mean', 'env_co2_std'
    ],
    'Audio': [
        'audio_amplitude_mean', 'audio_amplitude_max', 'audio_amplitude_std',
        'audio_freq_mean', 'audio_freq_max',
        'audio_centroid_mean', 'audio_centroid_std', 'audio_active_count'
    ],
    'Door': [
        'door_open_pct', 'door_max_duration', 'door_field_strength'
    ]
}

# Scenario descriptions for UI
SCENARIO_DESCRIPTIONS = {
    ScenarioType.NORMAL_ACTIVITY: "Regular daily activities with normal movement patterns",
    ScenarioType.NO_MOVEMENT: "Extended period without movement - possible sleep/rest",
    ScenarioType.PROLONGED_SILENCE: "Extended silence with minimal audio and motion",
    ScenarioType.FALL_DETECTED: "Fall signature followed by stillness",
    ScenarioType.MEDICAL_EMERGENCY: "Medical distress with abnormal vital signs",
    ScenarioType.DOOR_LEFT_OPEN: "Door left open for extended period",
    ScenarioType.WANDERING_RISK: "Door opened during quiet hours without caregiver",
    ScenarioType.ENTRAPMENT_RISK: "Person trapped in room/bathroom",
    ScenarioType.INTRUSION_ALERT: "Unauthorized entry while security system active",
    ScenarioType.ENVIRONMENTAL_HAZARD: "Environmental parameters beyond safe thresholds"
}

# Risk level descriptions and colors
RISK_LEVEL_INFO = {
    RiskLevel.LOW: {
        'description': 'Normal conditions, no immediate concern',
        'color': '#28a745',  # Green
        'priority': 1
    },
    RiskLevel.MEDIUM: {
        'description': 'Elevated risk, monitoring required',
        'color': '#ffc107',  # Yellow
        'priority': 2
    },
    RiskLevel.HIGH: {
        'description': 'High risk, caregiver attention needed',
        'color': '#fd7e14',  # Orange
        'priority': 3
    },
    RiskLevel.CRITICAL: {
        'description': 'Critical situation, immediate response required',
        'color': '#dc3545',  # Red
        'priority': 4
    }
}

# Communication settings
COMMUNICATION_CONFIG = {
    'alert_channels': ['sms', 'push_notification', 'email', 'phone_call'],
    'escalation_intervals_seconds': [30, 60, 300, 600],  # 30s, 1m, 5m, 10m
    'max_retry_attempts': 3,
    'emergency_services_number': '911',
    'backup_caregiver_timeout_seconds': 300
}

# Power management specifications
POWER_SPECIFICATIONS = {
    'total_system_power_w': 26.2,
    'sensor_array_power_w': 0.38,
    'battery_capacity_wh': 57.72,
    'target_runtime_hours': 24,
    'low_power_threshold_pct': 20,
    'critical_power_threshold_pct': 5
}

# Environmental operating limits
ENVIRONMENTAL_LIMITS = {
    'operating_temp_c': (-20, 60),
    'storage_temp_c': (-40, 85),
    'humidity_pct': (5, 95),
    'altitude_m': 3000,
    'vibration_g': 10,
    'ip_rating': 'IP65'
}

# Data retention policies
DATA_RETENTION = {
    'sensor_data_days': 30,
    'event_logs_days': 365,
    'model_versions_count': 10,
    'alert_history_days': 90,
    'performance_metrics_days': 180
}

# Edge deployment configurations
EDGE_DEPLOYMENT = {
    'raspberry_pi_5': {
        'target_inference_time_ms': 50,
        'max_model_size_mb': 100,
        'quantization': 'INT8',
        'optimization_level': 'O2'
    },
    'ai_hat_plus': {
        'target_inference_time_ms': 5,
        'max_model_size_mb': 500,
        'quantization': 'FP16',
        'optimization_level': 'O3'
    },
    'generic_edge': {
        'target_inference_time_ms': 100,
        'max_model_size_mb': 50,
        'quantization': 'INT8',
        'optimization_level': 'O1'
    }
}

