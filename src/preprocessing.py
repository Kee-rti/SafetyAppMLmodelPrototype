import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class SensorDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for multi-sensor fusion data
    """
    
    def __init__(
        self,
        normalization_method: str = "z-score",
        remove_outliers: bool = True,
        outlier_threshold: float = 1.5,
        window_size: int = 30,
        overlap: float = 0.5,
        feature_types: List[str] = None,
        sampling_rate: float = 1.0  # Hz
    ):
        """
        Initialize the preprocessing pipeline
        
        Args:
            normalization_method: Method for normalization ('z-score', 'min-max', 'robust', 'none')
            remove_outliers: Whether to remove outliers
            outlier_threshold: IQR multiplier for outlier detection
            window_size: Size of sliding window for feature extraction
            overlap: Overlap ratio between windows (0-1)
            feature_types: List of feature types to extract
            sampling_rate: Sampling rate in Hz
        """
        self.normalization_method = normalization_method
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        
        if feature_types is None:
            self.feature_types = ["statistical", "time_domain"]
        else:
            self.feature_types = feature_types
        
        # Initialize scalers
        self.scalers = {}
        self._init_scalers()
        
        # Preprocessing statistics
        self.preprocessing_stats = {
            'original_samples': 0,
            'processed_samples': 0,
            'feature_dimensions': 0,
            'outliers_removed': 0,
            'missing_values_imputed': 0
        }
    
    def _init_scalers(self):
        """Initialize normalization scalers"""
        if self.normalization_method == "z-score":
            self.main_scaler = StandardScaler()
        elif self.normalization_method == "min-max":
            self.main_scaler = MinMaxScaler()
        elif self.normalization_method == "robust":
            self.main_scaler = RobustScaler()
        else:
            self.main_scaler = None
    
    def preprocess_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Preprocess the entire dataset
        
        Args:
            dataset: List of dataset samples
            
        Returns:
            Dictionary containing processed data
        """
        self.preprocessing_stats['original_samples'] = len(dataset)
        
        # Extract features from all samples
        all_features = []
        all_labels = []
        all_metadata = []
        
        print("Extracting features from sensor data...")
        
        for i, sample in enumerate(dataset):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(dataset)}")
            
            try:
                # Extract features for this sample
                features = self._extract_sample_features(sample)
                
                if features is not None and len(features) > 0:
                    all_features.append(features)
                    
                    # Extract label
                    label = self._get_label(sample)
                    all_labels.append(label)
                    
                    # Extract metadata
                    metadata = {
                        'scenario': sample['scenario'],
                        'risk_level': sample['risk_level'],
                        'duration_h': sample['duration_h'],
                        'sample_id': sample.get('sample_id', f'sample_{i}')
                    }
                    all_metadata.append(metadata)
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid features extracted from dataset")
        
        # Convert to numpy arrays
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        print(f"Extracted features shape: {features_array.shape}")
        
        # Handle missing values
        features_array = self._handle_missing_values(features_array)
        
        # Remove outliers
        if self.remove_outliers:
            features_array, labels_array, all_metadata = self._remove_outliers(
                features_array, labels_array, all_metadata
            )
        
        # Normalize features
        if self.main_scaler is not None:
            features_array = self.main_scaler.fit_transform(features_array)
        
        # Create windowed sequences if requested
        if self.window_size > 1:
            windowed_features, windowed_labels, windowed_metadata = self._create_windows(
                features_array, labels_array, all_metadata
            )
        else:
            windowed_features = features_array
            windowed_labels = labels_array
            windowed_metadata = all_metadata
        
        # Update statistics
        self.preprocessing_stats['processed_samples'] = len(windowed_features)
        self.preprocessing_stats['feature_dimensions'] = windowed_features.shape[-1]
        
        processed_data = {
            'processed_features': windowed_features,
            'labels': windowed_labels,
            'metadata': windowed_metadata,
            'feature_names': self._get_feature_names(),
            'preprocessing_stats': self.preprocessing_stats
        }
        
        return processed_data
    
    def _extract_sample_features(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from a single sample"""
        try:
            sensor_readings = sample['sensor_readings']
            features = []
            
            # Process each sensor type
            for sensor_type, readings in sensor_readings.items():
                if not readings:
                    continue
                
                sensor_features = self._extract_sensor_features(sensor_type, readings)
                features.extend(sensor_features)
            
            return np.array(features, dtype=np.float32) if features else None
            
        except Exception as e:
            print(f"Error extracting features from sample: {e}")
            return None
    
    def _extract_sensor_features(self, sensor_type: str, readings: List) -> List[float]:
        """Extract features from specific sensor type"""
        features = []
        
        try:
            if sensor_type == 'pir':
                features.extend(self._extract_pir_features(readings))
            elif sensor_type == 'thermal':
                features.extend(self._extract_thermal_features(readings))
            elif sensor_type == 'radar':
                features.extend(self._extract_radar_features(readings))
            elif sensor_type == 'environmental':
                features.extend(self._extract_environmental_features(readings))
            elif sensor_type == 'audio':
                features.extend(self._extract_audio_features(readings))
            elif sensor_type == 'door':
                features.extend(self._extract_door_features(readings))
                
        except Exception as e:
            print(f"Error extracting {sensor_type} features: {e}")
            # Return zeros as fallback
            return [0.0] * self._get_sensor_feature_count(sensor_type)
        
        return features
    
    def _extract_pir_features(self, readings: List) -> List[float]:
        """Extract PIR sensor features"""
        if not readings:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        values = [float(r.value) if hasattr(r, 'value') else float(r) for r in readings]
        
        features = []
        
        # Statistical features
        if "statistical" in self.feature_types:
            features.extend([
                np.mean(values),
                np.std(values),
                np.max(values),
                np.min(values),
                np.median(values)
            ])
        
        # Time domain features
        if "time_domain" in self.feature_types:
            features.extend([
                np.sum(values),
                len([v for v in values if v > np.mean(values)]),  # Above average count
                np.var(values)
            ])
        
        # Frequency domain features
        if "frequency" in self.feature_types and len(values) > 4:
            try:
                fft_vals = np.abs(np.fft.fft(values))[:len(values)//2]
                features.extend([
                    np.max(fft_vals),
                    np.argmax(fft_vals),
                    np.mean(fft_vals)
                ])
            except:
                features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _extract_thermal_features(self, readings: List) -> List[float]:
        """Extract thermal sensor features"""
        if not readings:
            return [0.0] * 8
        
        features = []
        all_temps = []
        
        for r in readings:
            if hasattr(r, 'value') and isinstance(r.value, np.ndarray):
                all_temps.extend(r.value.flatten())
            elif hasattr(r, 'value') and isinstance(r.value, (list, tuple)):
                all_temps.extend(r.value)
            else:
                # Fallback for scalar values
                all_temps.append(float(r.value) if hasattr(r, 'value') else 22.0)
        
        if not all_temps:
            return [22.0] * 8  # Room temperature defaults
        
        all_temps = np.array(all_temps)
        
        # Statistical features
        if "statistical" in self.feature_types:
            features.extend([
                np.mean(all_temps),
                np.std(all_temps),
                np.max(all_temps),
                np.min(all_temps)
            ])
        
        # Thermal-specific features
        features.extend([
            np.percentile(all_temps, 95),  # Hot spots
            np.sum(all_temps > np.mean(all_temps) + 2 * np.std(all_temps)),  # Anomaly count
            np.var(all_temps),
            np.mean(all_temps) - np.min(all_temps)  # Temperature gradient
        ])
        
        return features
    
    def _extract_radar_features(self, readings: List) -> List[float]:
        """Extract radar sensor features"""
        if not readings:
            return [0.0] * 6
        
        ranges = []
        velocities = []
        strengths = []
        
        for r in readings:
            if hasattr(r, 'value') and isinstance(r.value, dict):
                ranges.append(r.value.get('range_m', 0.0))
                velocities.append(r.value.get('velocity_mps', 0.0))
                strengths.append(r.value.get('signal_strength', 0.0))
            else:
                ranges.append(0.0)
                velocities.append(0.0)
                strengths.append(0.0)
        
        features = []
        
        # Range features
        if ranges:
            features.extend([
                np.mean(ranges),
                np.std(ranges)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Velocity features
        if velocities:
            features.extend([
                np.mean(velocities),
                np.std(velocities)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Signal strength features
        if strengths:
            features.extend([
                np.mean(strengths),
                np.max(strengths)
            ])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_environmental_features(self, readings: List) -> List[float]:
        """Extract environmental sensor features"""
        if not readings:
            return [22.0, 45.0, 400.0, 0.0, 0.0, 0.0]
        
        temps = []
        humidity = []
        co2 = []
        
        for r in readings:
            if hasattr(r, 'value') and isinstance(r.value, dict):
                temps.append(r.value.get('temperature_c', 22.0))
                humidity.append(r.value.get('humidity_pct', 45.0))
                co2.append(r.value.get('co2_ppm', 400.0))
            else:
                temps.append(22.0)
                humidity.append(45.0)
                co2.append(400.0)
        
        features = []
        
        # Temperature features
        features.extend([
            np.mean(temps),
            np.std(temps) if len(temps) > 1 else 0.0
        ])
        
        # Humidity features
        features.extend([
            np.mean(humidity),
            np.std(humidity) if len(humidity) > 1 else 0.0
        ])
        
        # CO2 features
        features.extend([
            np.mean(co2),
            np.std(co2) if len(co2) > 1 else 0.0
        ])
        
        return features
    
    def _extract_audio_features(self, readings: List) -> List[float]:
        """Extract audio sensor features"""
        if not readings:
            return [0.0] * 8
        
        amplitudes = []
        frequencies = []
        centroids = []
        
        for r in readings:
            if hasattr(r, 'value') and isinstance(r.value, dict):
                amplitudes.append(r.value.get('amplitude_db', -60.0))
                frequencies.append(r.value.get('frequency_dominant_hz', 0.0))
                centroids.append(r.value.get('spectral_centroid', 0.0))
            else:
                amplitudes.append(-60.0)
                frequencies.append(0.0)
                centroids.append(0.0)
        
        features = []
        
        # Amplitude features
        features.extend([
            np.mean(amplitudes),
            np.max(amplitudes),
            np.std(amplitudes) if len(amplitudes) > 1 else 0.0
        ])
        
        # Frequency features
        features.extend([
            np.mean(frequencies),
            np.max(frequencies)
        ])
        
        # Spectral features
        features.extend([
            np.mean(centroids),
            np.std(centroids) if len(centroids) > 1 else 0.0,
            len([a for a in amplitudes if a > -40])  # Active audio count
        ])
        
        return features
    
    def _extract_door_features(self, readings: List) -> List[float]:
        """Extract door sensor features"""
        if not readings:
            return [0.0, 0.0, 1.0]
        
        open_states = []
        durations = []
        field_strengths = []
        
        for r in readings:
            if hasattr(r, 'value') and isinstance(r.value, dict):
                open_states.append(1.0 if r.value.get('is_open', False) else 0.0)
                durations.append(r.value.get('open_duration_s', 0.0))
                field_strengths.append(r.value.get('magnetic_field_strength', 1.0))
            else:
                open_states.append(0.0)
                durations.append(0.0)
                field_strengths.append(1.0)
        
        features = [
            np.mean(open_states),  # Percentage open
            np.max(durations),     # Max duration open
            np.mean(field_strengths)  # Average magnetic field
        ]
        
        return features
    
    def _get_sensor_feature_count(self, sensor_type: str) -> int:
        """Get expected feature count for sensor type"""
        feature_counts = {
            'pir': 8,      # 5 statistical + 3 time domain
            'thermal': 8,   # 4 statistical + 4 thermal-specific
            'radar': 6,     # 2 range + 2 velocity + 2 signal
            'environmental': 6,  # 2 temp + 2 humidity + 2 co2
            'audio': 8,     # 3 amplitude + 2 frequency + 3 spectral
            'door': 3       # open state + duration + field strength
        }
        
        base_count = feature_counts.get(sensor_type, 5)
        
        # Adjust based on feature types
        if "frequency" in self.feature_types and sensor_type == 'pir':
            base_count += 3
        
        return base_count
    
    def _get_label(self, sample: Dict[str, Any]) -> int:
        """Convert risk level to integer label"""
        risk_mapping = {
            'low': 0,
            'medium': 1, 
            'high': 2,
            'critical': 3
        }
        
        if hasattr(sample['risk_level'], 'value'):
            risk_str = sample['risk_level'].value
        else:
            risk_str = str(sample['risk_level']).lower()
        
        return risk_mapping.get(risk_str, 0)
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in features"""
        missing_count = np.sum(np.isnan(features))
        self.preprocessing_stats['missing_values_imputed'] = missing_count
        
        if missing_count > 0:
            # Simple imputation with column means
            col_means = np.nanmean(features, axis=0)
            inds = np.where(np.isnan(features))
            features[inds] = np.take(col_means, inds[1])
        
        return features
    
    def _remove_outliers(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        metadata: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Remove outliers using IQR method"""
        
        outlier_mask = np.ones(len(features), dtype=bool)
        
        for col in range(features.shape[1]):
            q1 = np.percentile(features[:, col], 25)
            q3 = np.percentile(features[:, col], 75)
            iqr = q3 - q1
            
            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr
            
            col_outliers = (features[:, col] < lower_bound) | (features[:, col] > upper_bound)
            outlier_mask &= ~col_outliers
        
        outliers_removed = np.sum(~outlier_mask)
        self.preprocessing_stats['outliers_removed'] = outliers_removed
        
        # Keep only non-outlier samples
        clean_features = features[outlier_mask]
        clean_labels = labels[outlier_mask]
        clean_metadata = [metadata[i] for i, keep in enumerate(outlier_mask) if keep]
        
        return clean_features, clean_labels, clean_metadata
    
    def _create_windows(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        metadata: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Create sliding windows from sequential data"""
        
        if len(features) < self.window_size:
            # If not enough data, just return as single window with padding
            padded_features = np.zeros((1, self.window_size, features.shape[1]))
            padded_features[0, :len(features)] = features
            return padded_features, labels[:1], metadata[:1]
        
        step_size = max(1, int(self.window_size * (1 - self.overlap)))
        
        windowed_features = []
        windowed_labels = []
        windowed_metadata = []
        
        for start in range(0, len(features) - self.window_size + 1, step_size):
            end = start + self.window_size
            
            # Extract window
            window_features = features[start:end]
            window_label = labels[start + self.window_size // 2]  # Use middle label
            window_meta = metadata[start + self.window_size // 2]  # Use middle metadata
            
            windowed_features.append(window_features)
            windowed_labels.append(window_label)
            windowed_metadata.append(window_meta)
        
        return np.array(windowed_features), np.array(windowed_labels), windowed_metadata
    
    def _get_feature_names(self) -> List[str]:
        """Generate feature names for interpretability"""
        feature_names = []
        
        # PIR features
        if "statistical" in self.feature_types:
            feature_names.extend([
                'pir_mean', 'pir_std', 'pir_max', 'pir_min', 'pir_median'
            ])
        if "time_domain" in self.feature_types:
            feature_names.extend([
                'pir_sum', 'pir_above_avg_count', 'pir_variance'
            ])
        if "frequency" in self.feature_types:
            feature_names.extend([
                'pir_fft_max', 'pir_fft_argmax', 'pir_fft_mean'
            ])
        
        # Thermal features
        feature_names.extend([
            'thermal_mean', 'thermal_std', 'thermal_max', 'thermal_min',
            'thermal_p95', 'thermal_anomaly_count', 'thermal_variance', 'thermal_gradient'
        ])
        
        # Radar features
        feature_names.extend([
            'radar_range_mean', 'radar_range_std',
            'radar_velocity_mean', 'radar_velocity_std',
            'radar_strength_mean', 'radar_strength_max'
        ])
        
        # Environmental features
        feature_names.extend([
            'env_temp_mean', 'env_temp_std',
            'env_humidity_mean', 'env_humidity_std',
            'env_co2_mean', 'env_co2_std'
        ])
        
        # Audio features
        feature_names.extend([
            'audio_amplitude_mean', 'audio_amplitude_max', 'audio_amplitude_std',
            'audio_freq_mean', 'audio_freq_max',
            'audio_centroid_mean', 'audio_centroid_std', 'audio_active_count'
        ])
        
        # Door features
        feature_names.extend([
            'door_open_pct', 'door_max_duration', 'door_field_strength'
        ])
        
        return feature_names
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        return self.preprocessing_stats.copy()
    
    def transform_new_sample(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        """Transform a new sample using fitted preprocessors"""
        try:
            # Extract features
            features = self._extract_sample_features(sample)
            if features is None:
                return None
            
            # Handle missing values
            features = features.reshape(1, -1)
            features = self._handle_missing_values(features)
            
            # Apply normalization
            if self.main_scaler is not None:
                features = self.main_scaler.transform(features)
            
            return features.flatten()
            
        except Exception as e:
            print(f"Error transforming new sample: {e}")
            return None

def create_feature_selector(method: str = 'kbest', k: int = 20) -> object:
    """
    Create feature selector for dimensionality reduction
    
    Args:
        method: Feature selection method
        k: Number of features to select
        
    Returns:
        Feature selector object
    """
    if method == 'kbest':
        return SelectKBest(score_func=f_classif, k=k)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

def apply_signal_filtering(
    signal_data: np.ndarray, 
    filter_type: str = 'lowpass',
    cutoff_freq: float = 0.1,
    sampling_rate: float = 1.0
) -> np.ndarray:
    """
    Apply signal filtering to sensor data
    
    Args:
        signal_data: Input signal
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
        cutoff_freq: Cutoff frequency
        sampling_rate: Sampling rate
        
    Returns:
        Filtered signal
    """
    try:
        nyquist = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        if filter_type == 'lowpass':
            b, a = signal.butter(4, normalized_cutoff, btype='low')
        elif filter_type == 'highpass':
            b, a = signal.butter(4, normalized_cutoff, btype='high')
        else:
            return signal_data  # No filtering
        
        filtered_signal = signal.filtfilt(b, a, signal_data)
        return filtered_signal
        
    except Exception as e:
        print(f"Error applying signal filter: {e}")
        return signal_data

