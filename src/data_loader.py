import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader

# Add the path to find the dataset generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from attached_assets.safetyAppDataset_1755171350672 import TRSCompliantDatasetGenerator, ScenarioType, RiskLevel, SensorReading
except ImportError:
    # Fallback imports in case the file path is different
    from utils.constants import ScenarioType, RiskLevel
    
    # Define SensorReading as a fallback
    from dataclasses import dataclass
    
    @dataclass
    class SensorReading:
        sensor_id: str
        sensor_type: str
        value: Any
        confidence: float
        timestamp: float
        metadata: Dict[str, Any]
        power_consumption_mw: float = 0.0
        sensor_health: float = 1.0
        calibration_drift: float = 0.0

class TRSDataLoader:
    """
    Data loader for TRS-compliant sensor fusion dataset
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data loader with TRS-compliant dataset generator
        
        Args:
            seed: Random seed for reproducible dataset generation
        """
        self.seed = seed
        np.random.seed(seed)
        try:
            self.generator = TRSCompliantDatasetGenerator(seed=seed)
        except Exception as e:
            print(f"Warning: Could not initialize TRSCompliantDatasetGenerator: {e}")
            self.generator = None
    
    def generate_comprehensive_dataset(
        self, 
        num_scenarios: int = 1000,
        duration_range: Tuple[float, float] = (1.0, 8.0),
        scenario_types: List[ScenarioType] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a comprehensive TRS-compliant dataset
        
        Args:
            num_scenarios: Number of scenarios to generate
            duration_range: Range of scenario durations in hours
            scenario_types: List of scenario types to include
            
        Returns:
            List of dataset samples
        """
        if scenario_types is None:
            scenario_types = [
                ScenarioType.NORMAL_ACTIVITY,
                ScenarioType.NO_MOVEMENT,
                ScenarioType.FALL_DETECTED,
                ScenarioType.MEDICAL_EMERGENCY,
                ScenarioType.PROLONGED_SILENCE,
                ScenarioType.DOOR_LEFT_OPEN,
                ScenarioType.WANDERING_RISK,
                ScenarioType.ENVIRONMENTAL_HAZARD
            ]
        
        dataset = []
        
        if self.generator is None:
            # Generate synthetic dataset if generator not available
            return self._generate_synthetic_dataset(num_scenarios, duration_range, scenario_types)
        
        try:
            # Use the actual TRS generator if available
            for i in range(num_scenarios):
                scenario = np.random.choice(scenario_types)
                duration = np.random.uniform(*duration_range)
                
                # Generate sensor readings for this scenario
                sensor_data = self._generate_scenario_data(scenario, duration)
                
                # Create sample
                sample = {
                    'scenario': scenario,
                    'risk_level': self._get_risk_level(scenario),
                    'duration_h': duration,
                    'sensor_readings': sensor_data,
                    'timestamp': datetime.now().timestamp(),
                    'sample_id': f"sample_{i:06d}"
                }
                
                dataset.append(sample)
            
            return dataset
            
        except Exception as e:
            print(f"Error generating dataset: {e}")
            return self._generate_synthetic_dataset(num_scenarios, duration_range, scenario_types)
    
    def _generate_scenario_data(self, scenario: ScenarioType, duration_h: float) -> Dict[str, List[SensorReading]]:
        """
        Generate sensor readings for a specific scenario
        
        Args:
            scenario: The scenario type
            duration_h: Duration in hours
            
        Returns:
            Dictionary of sensor readings by type
        """
        sensor_data = {
            'pir': [],
            'thermal': [],
            'radar': [],
            'environmental': [],
            'audio': [],
            'door': []
        }
        
        # Generate time series data
        num_samples = int(duration_h * 60)  # One sample per minute
        timestamps = np.linspace(0, duration_h * 3600, num_samples)
        
        for i, timestamp in enumerate(timestamps):
            # PIR sensor data
            pir_value = self._generate_pir_data(scenario, i, num_samples)
            sensor_data['pir'].append(SensorReading(
                sensor_id="pir_001",
                sensor_type="pir",
                value=pir_value,
                confidence=np.random.uniform(0.8, 1.0),
                timestamp=timestamp,
                metadata={"detection_zone": "main_area"},
                power_consumption_mw=0.165
            ))
            
            # Thermal sensor data
            thermal_value = self._generate_thermal_data(scenario, i, num_samples)
            sensor_data['thermal'].append(SensorReading(
                sensor_id="thermal_001",
                sensor_type="thermal",
                value=thermal_value,
                confidence=np.random.uniform(0.85, 1.0),
                timestamp=timestamp,
                metadata={"resolution": "8x8", "fov": "60x60"},
                power_consumption_mw=14.85
            ))
            
            # Radar sensor data
            radar_value = self._generate_radar_data(scenario, i, num_samples)
            sensor_data['radar'].append(SensorReading(
                sensor_id="radar_001",
                sensor_type="radar",
                value=radar_value,
                confidence=np.random.uniform(0.9, 1.0),
                timestamp=timestamp,
                metadata={"frequency_ghz": 60, "range_m": 10},
                power_consumption_mw=280.5
            ))
            
            # Environmental sensors
            env_value = self._generate_environmental_data(scenario, i, num_samples)
            sensor_data['environmental'].append(SensorReading(
                sensor_id="env_001",
                sensor_type="environmental",
                value=env_value,
                confidence=np.random.uniform(0.95, 1.0),
                timestamp=timestamp,
                metadata={"sensors": ["temperature", "humidity", "co2"]},
                power_consumption_mw=65.5
            ))
            
            # Audio sensor data
            audio_value = self._generate_audio_data(scenario, i, num_samples)
            sensor_data['audio'].append(SensorReading(
                sensor_id="audio_001",
                sensor_type="audio",
                value=audio_value,
                confidence=np.random.uniform(0.7, 0.95),
                timestamp=timestamp,
                metadata={"frequency_range": "60-15000Hz", "snr_db": 61},
                power_consumption_mw=4.62
            ))
            
            # Door sensor data
            door_value = self._generate_door_data(scenario, i, num_samples)
            sensor_data['door'].append(SensorReading(
                sensor_id="door_001",
                sensor_type="door",
                value=door_value,
                confidence=np.random.uniform(0.98, 1.0),
                timestamp=timestamp,
                metadata={"sensor_type": "magnetic", "gap_distance_mm": 20},
                power_consumption_mw=0.0
            ))
        
        return sensor_data
    
    def _generate_pir_data(self, scenario: ScenarioType, sample_idx: int, total_samples: int) -> float:
        """Generate PIR sensor data based on scenario"""
        base_movement = 0.1
        
        if scenario == ScenarioType.NORMAL_ACTIVITY:
            return base_movement + 0.6 * np.random.random() + 0.2 * np.sin(2 * np.pi * sample_idx / 60)
        elif scenario == ScenarioType.NO_MOVEMENT:
            return base_movement * np.random.random()
        elif scenario == ScenarioType.FALL_DETECTED:
            if sample_idx < total_samples * 0.1:  # Fall event
                return 0.9 + 0.1 * np.random.random()
            else:  # After fall - no movement
                return base_movement * 0.1 * np.random.random()
        elif scenario == ScenarioType.MEDICAL_EMERGENCY:
            return base_movement + 0.3 * np.random.random() + 0.1 * np.sin(10 * np.pi * sample_idx / total_samples)
        elif scenario == ScenarioType.WANDERING_RISK:
            return 0.4 + 0.4 * np.random.random() + 0.2 * np.sin(4 * np.pi * sample_idx / total_samples)
        else:
            return base_movement + 0.3 * np.random.random()
    
    def _generate_thermal_data(self, scenario: ScenarioType, sample_idx: int, total_samples: int) -> np.ndarray:
        """Generate thermal sensor data (8x8 grid)"""
        base_temp = 22.0  # Room temperature
        thermal_grid = np.full((8, 8), base_temp) + np.random.normal(0, 0.5, (8, 8))
        
        if scenario == ScenarioType.NORMAL_ACTIVITY:
            # Human presence signature
            center_x, center_y = 4, 4
            human_temp = 36.0 + np.random.normal(0, 1.0)
            thermal_grid[center_x-1:center_x+2, center_y-1:center_y+2] = human_temp
            
        elif scenario == ScenarioType.NO_MOVEMENT:
            # Stationary human
            center_x, center_y = 4, 4
            human_temp = 35.0 + np.random.normal(0, 0.5)
            thermal_grid[center_x-1:center_x+2, center_y-1:center_y+2] = human_temp
            
        elif scenario == ScenarioType.FALL_DETECTED:
            # Fallen person (horizontal signature)
            if sample_idx > total_samples * 0.1:
                human_temp = 35.5 + np.random.normal(0, 0.5)
                thermal_grid[3:6, 2:7] = human_temp  # Horizontal pattern
                
        elif scenario == ScenarioType.ENVIRONMENTAL_HAZARD:
            # Temperature anomaly
            thermal_grid += np.random.uniform(5, 15)
        
        return thermal_grid.flatten()
    
    def _generate_radar_data(self, scenario: ScenarioType, sample_idx: int, total_samples: int) -> Dict[str, float]:
        """Generate radar sensor data"""
        radar_data = {
            'range_m': 0.0,
            'velocity_mps': 0.0,
            'signal_strength': 0.0
        }
        
        if scenario == ScenarioType.NORMAL_ACTIVITY:
            radar_data['range_m'] = 2.0 + 3.0 * np.random.random()
            radar_data['velocity_mps'] = np.random.uniform(-2.0, 2.0)
            radar_data['signal_strength'] = 0.7 + 0.3 * np.random.random()
            
        elif scenario == ScenarioType.NO_MOVEMENT:
            radar_data['range_m'] = 2.5 + 0.5 * np.random.random()
            radar_data['velocity_mps'] = np.random.uniform(-0.1, 0.1)
            radar_data['signal_strength'] = 0.4 + 0.2 * np.random.random()
            
        elif scenario == ScenarioType.FALL_DETECTED:
            if sample_idx < total_samples * 0.1:  # During fall
                radar_data['range_m'] = 2.0 + np.random.random()
                radar_data['velocity_mps'] = np.random.uniform(-5.0, -2.0)  # Downward motion
                radar_data['signal_strength'] = 0.9 + 0.1 * np.random.random()
            else:  # After fall
                radar_data['range_m'] = 1.5 + 0.5 * np.random.random()
                radar_data['velocity_mps'] = np.random.uniform(-0.05, 0.05)
                radar_data['signal_strength'] = 0.6 + 0.2 * np.random.random()
        
        return radar_data
    
    def _generate_environmental_data(self, scenario: ScenarioType, sample_idx: int, total_samples: int) -> Dict[str, float]:
        """Generate environmental sensor data"""
        env_data = {
            'temperature_c': 22.0 + np.random.normal(0, 1.0),
            'humidity_pct': 45.0 + np.random.normal(0, 5.0),
            'co2_ppm': 400.0 + np.random.normal(0, 50.0)
        }
        
        if scenario == ScenarioType.ENVIRONMENTAL_HAZARD:
            env_data['temperature_c'] += np.random.uniform(10, 25)
            env_data['humidity_pct'] += np.random.uniform(-20, 30)
            env_data['co2_ppm'] += np.random.uniform(1000, 3000)
            
        elif scenario == ScenarioType.NORMAL_ACTIVITY:
            # Human presence affects CO2 and humidity
            env_data['co2_ppm'] += np.random.uniform(100, 300)
            env_data['humidity_pct'] += np.random.uniform(5, 15)
        
        return env_data
    
    def _generate_audio_data(self, scenario: ScenarioType, sample_idx: int, total_samples: int) -> Dict[str, float]:
        """Generate audio sensor data"""
        audio_data = {
            'amplitude_db': -60.0 + np.random.uniform(0, 20),  # Background noise
            'frequency_dominant_hz': 0.0,
            'spectral_centroid': 0.0,
            'mfcc_features': np.random.normal(0, 1, 13).tolist()  # 13 MFCC coefficients
        }
        
        if scenario == ScenarioType.NORMAL_ACTIVITY:
            audio_data['amplitude_db'] += np.random.uniform(10, 30)
            audio_data['frequency_dominant_hz'] = np.random.uniform(200, 4000)
            audio_data['spectral_centroid'] = np.random.uniform(1000, 3000)
            
        elif scenario == ScenarioType.PROLONGED_SILENCE:
            audio_data['amplitude_db'] = -60.0 + np.random.uniform(0, 5)
            
        elif scenario == ScenarioType.FALL_DETECTED:
            if sample_idx < total_samples * 0.05:  # Impact sound
                audio_data['amplitude_db'] = -20.0 + np.random.uniform(0, 10)
                audio_data['frequency_dominant_hz'] = np.random.uniform(100, 500)
                
        elif scenario == ScenarioType.MEDICAL_EMERGENCY:
            # Distressed sounds
            audio_data['amplitude_db'] += np.random.uniform(5, 25)
            audio_data['frequency_dominant_hz'] = np.random.uniform(300, 1000)
        
        return audio_data
    
    def _generate_door_data(self, scenario: ScenarioType, sample_idx: int, total_samples: int) -> Dict[str, Any]:
        """Generate door sensor data"""
        door_data = {
            'is_open': False,
            'open_duration_s': 0.0,
            'magnetic_field_strength': 1.0  # Closed state
        }
        
        if scenario == ScenarioType.DOOR_LEFT_OPEN:
            door_data['is_open'] = True
            door_data['open_duration_s'] = sample_idx * 60.0  # Minutes to seconds
            door_data['magnetic_field_strength'] = 0.1
            
        elif scenario == ScenarioType.WANDERING_RISK:
            # Door opened during quiet hours
            if sample_idx < total_samples * 0.2:
                door_data['is_open'] = True
                door_data['open_duration_s'] = sample_idx * 60.0
                door_data['magnetic_field_strength'] = 0.1
            else:
                door_data['is_open'] = False
                door_data['magnetic_field_strength'] = 1.0
                
        elif scenario == ScenarioType.INTRUSION_ALERT:
            # Unexpected door opening
            if np.random.random() < 0.3:  # 30% chance of door activity
                door_data['is_open'] = True
                door_data['open_duration_s'] = np.random.uniform(5, 60)
                door_data['magnetic_field_strength'] = 0.1
        
        return door_data
    
    def _get_risk_level(self, scenario: ScenarioType) -> RiskLevel:
        """Map scenario to risk level"""
        risk_mapping = {
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
        return risk_mapping.get(scenario, RiskLevel.LOW)
    
    def _generate_synthetic_dataset(
        self, 
        num_scenarios: int, 
        duration_range: Tuple[float, float], 
        scenario_types: List[ScenarioType]
    ) -> List[Dict[str, Any]]:
        """
        Fallback method to generate synthetic dataset when TRS generator is not available
        """
        dataset = []
        
        for i in range(num_scenarios):
            scenario = np.random.choice(scenario_types)
            duration = np.random.uniform(*duration_range)
            
            # Generate synthetic sensor data
            sensor_data = self._generate_scenario_data(scenario, duration)
            
            sample = {
                'scenario': scenario,
                'risk_level': self._get_risk_level(scenario),
                'duration_h': duration,
                'sensor_readings': sensor_data,
                'timestamp': datetime.now().timestamp(),
                'sample_id': f"synthetic_sample_{i:06d}"
            }
            
            dataset.append(sample)
        
        return dataset
    
    def get_dataset_statistics(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculate dataset statistics
        
        Args:
            dataset: List of dataset samples
            
        Returns:
            DataFrame with statistics
        """
        if not dataset:
            return pd.DataFrame()
        
        # Extract basic statistics
        scenarios = [sample['scenario'].value for sample in dataset]
        risk_levels = [sample['risk_level'].value for sample in dataset]
        durations = [sample['duration_h'] for sample in dataset]
        
        # Count distributions
        scenario_counts = pd.Series(scenarios).value_counts()
        risk_counts = pd.Series(risk_levels).value_counts()
        
        # Create statistics dataframe
        stats_data = []
        
        # Scenario statistics
        for scenario, count in scenario_counts.items():
            stats_data.append({
                'Category': 'Scenario',
                'Type': scenario,
                'Count': count,
                'Percentage': f"{count/len(dataset)*100:.1f}%"
            })
        
        # Risk level statistics
        for risk, count in risk_counts.items():
            stats_data.append({
                'Category': 'Risk Level',
                'Type': risk,
                'Count': count,
                'Percentage': f"{count/len(dataset)*100:.1f}%"
            })
        
        # Duration statistics
        stats_data.append({
            'Category': 'Duration',
            'Type': 'Mean (hours)',
            'Count': f"{np.mean(durations):.2f}",
            'Percentage': '-'
        })
        
        stats_data.append({
            'Category': 'Duration',
            'Type': 'Std (hours)',
            'Count': f"{np.std(durations):.2f}",
            'Percentage': '-'
        })
        
        return pd.DataFrame(stats_data)

class SensorFusionDataset(Dataset):
    """
    PyTorch Dataset for sensor fusion data
    """
    
    def __init__(self, dataset: List[Dict[str, Any]], transform=None):
        """
        Initialize the PyTorch dataset
        
        Args:
            dataset: List of dataset samples
            transform: Optional transform to apply to data
        """
        self.dataset = dataset
        self.transform = transform
        
        # Extract features and labels
        self.features, self.labels = self._prepare_data()
    
    def _prepare_data(self) -> Tuple[List[torch.Tensor], List[int]]:
        """Prepare features and labels for training"""
        features = []
        labels = []
        
        # Risk level to integer mapping
        risk_to_int = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3
        }
        
        for sample in self.dataset:
            # Convert sensor readings to feature vectors
            feature_vector = self._extract_features(sample['sensor_readings'])
            features.append(torch.FloatTensor(feature_vector))
            
            # Convert risk level to integer
            labels.append(risk_to_int[sample['risk_level']])
        
        return features, labels
    
    def _extract_features(self, sensor_readings: Dict[str, List[SensorReading]]) -> np.ndarray:
        """Extract numerical features from sensor readings"""
        features = []
        
        for sensor_type, readings in sensor_readings.items():
            if not readings:
                continue
                
            if sensor_type == 'pir':
                # PIR features: mean, std, max, min, median, sum, above_avg_count, variance
                values = [float(r.value) if hasattr(r, 'value') else float(r) for r in readings]
                if values:
                    features.extend([
                        np.mean(values), np.std(values), np.max(values), np.min(values),
                        np.median(values), np.sum(values), 
                        len([v for v in values if v > np.mean(values)]),
                        np.var(values)
                    ])
                else:
                    features.extend([0.0] * 8)
                
            elif sensor_type == 'thermal':
                # Thermal features: mean, std, max, min, p95, anomaly_count, variance, gradient
                thermal_data = []
                for r in readings:
                    if hasattr(r, 'value') and isinstance(r.value, np.ndarray):
                        thermal_data.extend(r.value.flatten())
                    elif hasattr(r, 'value') and isinstance(r.value, (list, tuple)):
                        thermal_data.extend(r.value)
                    else:
                        thermal_data.append(22.0)  # Default room temperature
                
                if thermal_data:
                    thermal_array = np.array(thermal_data)
                    features.extend([
                        np.mean(thermal_array), np.std(thermal_array), np.max(thermal_array), np.min(thermal_array),
                        np.percentile(thermal_array, 95), 
                        np.sum(thermal_array > np.mean(thermal_array) + 2 * np.std(thermal_array)),
                        np.var(thermal_array), np.mean(thermal_array) - np.min(thermal_array)
                    ])
                else:
                    features.extend([22.0, 1.0, 25.0, 20.0, 24.0, 0.0, 1.0, 2.0])
                    
            elif sensor_type == 'radar':
                # Radar features: range_mean, range_std, velocity_mean, velocity_std, strength_mean, strength_max
                ranges, velocities, strengths = [], [], []
                for r in readings:
                    if hasattr(r, 'value') and isinstance(r.value, dict):
                        ranges.append(r.value.get('range_m', 0.0))
                        velocities.append(r.value.get('velocity_mps', 0.0))
                        strengths.append(r.value.get('signal_strength', 0.0))
                    else:
                        ranges.append(0.0)
                        velocities.append(0.0)
                        strengths.append(0.0)
                
                features.extend([
                    np.mean(ranges) if ranges else 0.0, np.std(ranges) if ranges else 0.0,
                    np.mean(velocities) if velocities else 0.0, np.std(velocities) if velocities else 0.0,
                    np.mean(strengths) if strengths else 0.0, np.max(strengths) if strengths else 0.0
                ])
                    
            elif sensor_type == 'environmental':
                # Environmental features: temp_mean, temp_std, humidity_mean, humidity_std, co2_mean, co2_std
                temps, humidity, co2 = [], [], []
                for r in readings:
                    if hasattr(r, 'value') and isinstance(r.value, dict):
                        temps.append(r.value.get('temperature_c', 22.0))
                        humidity.append(r.value.get('humidity_pct', 45.0))
                        co2.append(r.value.get('co2_ppm', 400.0))
                    else:
                        temps.append(22.0)
                        humidity.append(45.0)
                        co2.append(400.0)
                
                features.extend([
                    np.mean(temps) if temps else 22.0, np.std(temps) if temps else 0.0,
                    np.mean(humidity) if humidity else 45.0, np.std(humidity) if humidity else 0.0,
                    np.mean(co2) if co2 else 400.0, np.std(co2) if co2 else 0.0
                ])
                    
            elif sensor_type == 'audio':
                # Audio features: amplitude_mean, amplitude_max, amplitude_std, freq_mean, freq_max, centroid_mean, centroid_std, active_count
                amplitudes, frequencies, centroids = [], [], []
                for r in readings:
                    if hasattr(r, 'value') and isinstance(r.value, dict):
                        amplitudes.append(r.value.get('amplitude_db', -60.0))
                        frequencies.append(r.value.get('frequency_dominant_hz', 0.0))
                        centroids.append(r.value.get('spectral_centroid', 0.0))
                    else:
                        amplitudes.append(-60.0)
                        frequencies.append(0.0)
                        centroids.append(0.0)
                
                features.extend([
                    np.mean(amplitudes) if amplitudes else -60.0,
                    np.max(amplitudes) if amplitudes else -60.0,
                    np.std(amplitudes) if amplitudes else 0.0,
                    np.mean(frequencies) if frequencies else 0.0,
                    np.max(frequencies) if frequencies else 0.0,
                    np.mean(centroids) if centroids else 0.0,
                    np.std(centroids) if centroids else 0.0,
                    len([a for a in amplitudes if a > -40]) if amplitudes else 0
                ])
                    
            elif sensor_type == 'door':
                # Door features: open_pct, max_duration, field_strength
                open_states, durations, field_strengths = [], [], []
                for r in readings:
                    if hasattr(r, 'value') and isinstance(r.value, dict):
                        open_states.append(1.0 if r.value.get('is_open', False) else 0.0)
                        durations.append(r.value.get('open_duration_s', 0.0))
                        field_strengths.append(r.value.get('magnetic_field_strength', 1.0))
                    else:
                        open_states.append(0.0)
                        durations.append(0.0)
                        field_strengths.append(1.0)
                
                features.extend([
                    np.mean(open_states) if open_states else 0.0,
                    np.max(durations) if durations else 0.0,
                    np.mean(field_strengths) if field_strengths else 1.0
                ])
        
        # Ensure consistent feature size (39 total features)
        target_size = 39
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return np.array(features, dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label

def create_data_loaders(
    dataset: List[Dict[str, Any]], 
    batch_size: int = 32, 
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders for training, validation, and testing
    
    Args:
        dataset: Complete dataset
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Shuffle dataset
    np.random.shuffle(dataset)
    
    # Calculate split indices
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    # Split dataset
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    # Create datasets
    train_dataset = SensorFusionDataset(train_data)
    val_dataset = SensorFusionDataset(val_data)
    test_dataset = SensorFusionDataset(test_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
