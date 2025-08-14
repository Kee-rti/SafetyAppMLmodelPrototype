"""
Sensor specifications based on TRS documentation for multi-sensor fusion safety monitoring
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from utils.constants import SENSOR_SPECIFICATIONS, ScenarioType, RiskLevel

def get_sensor_specifications(sensor_type: str) -> Dict[str, Any]:
    """
    Get detailed sensor specifications based on TRS documentation
    
    Args:
        sensor_type: Type of sensor (PIR, Thermal, Radar, Environmental, Audio, Door)
        
    Returns:
        Dictionary containing sensor specifications
    """
    sensor_type = sensor_type.upper()
    
    if sensor_type not in SENSOR_SPECIFICATIONS:
        raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    return SENSOR_SPECIFICATIONS[sensor_type]

def get_sensor_power_consumption(sensor_type: str) -> float:
    """
    Get power consumption for specific sensor type in milliwatts
    
    Args:
        sensor_type: Type of sensor
        
    Returns:
        Power consumption in mW
    """
    specs = get_sensor_specifications(sensor_type)
    
    if sensor_type.upper() == 'ENVIRONMENTAL':
        # Sum up all environmental sensors
        total_power = 0.0
        for env_sensor in specs.values():
            if isinstance(env_sensor, dict) and 'power_consumption_mw' in env_sensor:
                total_power += env_sensor['power_consumption_mw']
        return total_power
    elif sensor_type.upper() == 'DOOR':
        # Use hall effect sensor power (magnetic sensor uses no power)
        return specs['hall_effect']['power_consumption_mw']
    else:
        return specs.get('power_consumption_mw', 0.0)

def get_sensor_operating_ranges(sensor_type: str) -> Dict[str, Tuple[float, float]]:
    """
    Get operating ranges for sensor parameters
    
    Args:
        sensor_type: Type of sensor
        
    Returns:
        Dictionary of parameter ranges
    """
    specs = get_sensor_specifications(sensor_type)
    ranges = {}
    
    if sensor_type.upper() == 'PIR':
        ranges = {
            'detection_range_m': specs['detection_range_m'],
            'operating_voltage': specs['operating_voltage'],
            'delay_time_s': (0.3, 18.0),
            'operating_temp_c': (-15, 70)
        }
    elif sensor_type.upper() == 'THERMAL':
        ranges = {
            'temp_range_c': specs['temp_range_c'],
            'field_of_view_deg': specs['field_of_view_deg'],
            'accuracy_c': (0, specs['accuracy_c']),
            'refresh_rate_hz': (1, specs['refresh_rate_hz'])
        }
    elif sensor_type.upper() == 'RADAR':
        ranges = {
            'frequency_range_ghz': specs['frequency_range_ghz'],
            'detection_range_m': specs['detection_range_m'],
            'velocity_range_mps': specs['velocity_range_mps'],
            'range_resolution_cm': (5, specs['range_resolution_cm'])
        }
    elif sensor_type.upper() == 'ENVIRONMENTAL':
        ranges = {
            'temperature_range_c': specs['temperature']['range_c'],
            'humidity_range_pct': specs['humidity']['range_pct'],
            'co2_range_ppm': specs['co2']['range_ppm'],
            'temp_accuracy_c': (0, specs['temperature']['accuracy_c']),
            'humidity_accuracy_pct': (0, specs['humidity']['accuracy_pct'])
        }
    elif sensor_type.upper() == 'AUDIO':
        ranges = {
            'frequency_response_hz': specs['frequency_response_hz'],
            'acoustic_overload_point_db': (0, specs['acoustic_overload_point_db']),
            'snr_db': (0, specs['snr_db']),
            'sensitivity_dbfs': (specs['sensitivity_dbfs'], 0)
        }
    elif sensor_type.upper() == 'DOOR':
        ranges = {
            'gap_distance_mm': (0, specs['magnetic']['gap_distance_mm']),
            'magnetic_threshold_gauss': (specs['hall_effect']['magnetic_threshold_gauss']['release'],
                                       specs['hall_effect']['magnetic_threshold_gauss']['operate']),
            'operating_temp_c': (-40, 85)
        }
    
    return ranges

def validate_sensor_reading(sensor_type: str, reading_value: Any) -> bool:
    """
    Validate if a sensor reading is within expected ranges
    
    Args:
        sensor_type: Type of sensor
        reading_value: The reading value to validate
        
    Returns:
        True if reading is valid, False otherwise
    """
    try:
        ranges = get_sensor_operating_ranges(sensor_type)
        
        if sensor_type.upper() == 'PIR':
            # PIR readings should be 0-1 (normalized)
            return 0 <= float(reading_value) <= 1
            
        elif sensor_type.upper() == 'THERMAL':
            # Thermal readings are temperature arrays
            if isinstance(reading_value, (list, np.ndarray)):
                temps = np.array(reading_value).flatten()
                temp_range = ranges['temp_range_c']
                return np.all((temp_range[0] <= temps) & (temps <= temp_range[1]))
            return False
            
        elif sensor_type.upper() == 'RADAR':
            # Radar readings are dictionaries
            if isinstance(reading_value, dict):
                range_m = reading_value.get('range_m', 0)
                velocity_mps = reading_value.get('velocity_mps', 0)
                
                range_valid = ranges['detection_range_m'][0] <= range_m <= ranges['detection_range_m'][1]
                velocity_valid = ranges['velocity_range_mps'][0] <= velocity_mps <= ranges['velocity_range_mps'][1]
                
                return range_valid and velocity_valid
            return False
            
        elif sensor_type.upper() == 'ENVIRONMENTAL':
            # Environmental readings are dictionaries
            if isinstance(reading_value, dict):
                temp = reading_value.get('temperature_c', 22)
                humidity = reading_value.get('humidity_pct', 45)
                co2 = reading_value.get('co2_ppm', 400)
                
                temp_valid = ranges['temperature_range_c'][0] <= temp <= ranges['temperature_range_c'][1]
                humidity_valid = ranges['humidity_range_pct'][0] <= humidity <= ranges['humidity_range_pct'][1]
                co2_valid = ranges['co2_range_ppm'][0] <= co2 <= ranges['co2_range_ppm'][1]
                
                return temp_valid and humidity_valid and co2_valid
            return False
            
        elif sensor_type.upper() == 'AUDIO':
            # Audio readings are dictionaries
            if isinstance(reading_value, dict):
                amplitude = reading_value.get('amplitude_db', -60)
                frequency = reading_value.get('frequency_dominant_hz', 0)
                
                freq_range = ranges['frequency_response_hz']
                freq_valid = freq_range[0] <= frequency <= freq_range[1] if frequency > 0 else True
                amplitude_valid = -120 <= amplitude <= ranges['acoustic_overload_point_db'][1]
                
                return freq_valid and amplitude_valid
            return False
            
        elif sensor_type.upper() == 'DOOR':
            # Door readings are dictionaries
            if isinstance(reading_value, dict):
                is_open = reading_value.get('is_open', False)
                duration = reading_value.get('open_duration_s', 0)
                field_strength = reading_value.get('magnetic_field_strength', 1.0)
                
                duration_valid = duration >= 0
                field_valid = 0 <= field_strength <= 1.5
                
                return duration_valid and field_valid
            return False
            
        return False
        
    except Exception:
        return False

def get_sensor_feature_dimensions() -> Dict[str, int]:
    """
    Get the number of features extracted from each sensor type
    
    Returns:
        Dictionary mapping sensor types to feature dimensions
    """
    return {
        'PIR': 8,      # Statistical + time domain features
        'Thermal': 8,   # Thermal-specific features
        'Radar': 6,     # Range, velocity, signal strength features
        'Environmental': 6,  # Temperature, humidity, CO2 features
        'Audio': 8,     # Amplitude, frequency, spectral features
        'Door': 3       # Open state, duration, field strength
    }

def get_total_feature_dimensions() -> int:
    """
    Get total number of features across all sensors
    
    Returns:
        Total feature dimension count
    """
    return sum(get_sensor_feature_dimensions().values())

def get_sensor_data_quality_thresholds() -> Dict[str, Dict[str, float]]:
    """
    Get data quality thresholds for each sensor type
    
    Returns:
        Dictionary of quality thresholds
    """
    return {
        'PIR': {
            'min_confidence': 0.8,
            'max_noise_level': 0.1,
            'min_detection_rate': 0.95
        },
        'Thermal': {
            'min_confidence': 0.85,
            'max_temperature_drift': 2.0,
            'min_spatial_resolution': 64  # 8x8 grid
        },
        'Radar': {
            'min_confidence': 0.9,
            'max_range_error': 0.15,  # 15cm
            'max_velocity_error': 0.2  # 0.2 m/s
        },
        'Environmental': {
            'min_confidence': 0.95,
            'max_temp_drift': 0.5,
            'max_humidity_drift': 1.8,
            'max_co2_drift': 50
        },
        'Audio': {
            'min_confidence': 0.7,
            'min_snr_db': 40,
            'max_distortion_thd': 0.1
        },
        'Door': {
            'min_confidence': 0.98,
            'max_gap_distance_mm': 20,
            'min_magnetic_field_strength': 0.1
        }
    }

def calculate_sensor_reliability_score(sensor_type: str, readings: List[Any]) -> float:
    """
    Calculate reliability score for sensor based on readings
    
    Args:
        sensor_type: Type of sensor
        readings: List of sensor readings
        
    Returns:
        Reliability score between 0 and 1
    """
    if not readings:
        return 0.0
    
    try:
        # Count valid readings
        valid_count = sum(1 for reading in readings if validate_sensor_reading(sensor_type, reading))
        validity_score = valid_count / len(readings)
        
        # Get quality thresholds
        thresholds = get_sensor_data_quality_thresholds().get(sensor_type.upper(), {})
        
        # Calculate consistency score
        if sensor_type.upper() == 'PIR':
            values = [float(r) if not isinstance(r, dict) else r.get('value', 0) for r in readings]
            consistency_score = 1.0 - min(np.std(values), 0.5) / 0.5
            
        elif sensor_type.upper() == 'THERMAL':
            # Thermal consistency based on temperature stability
            temps = []
            for r in readings:
                if isinstance(r, dict) and 'value' in r:
                    if isinstance(r['value'], (list, np.ndarray)):
                        temps.extend(np.array(r['value']).flatten())
            
            if temps:
                temp_std = np.std(temps)
                consistency_score = max(0, 1.0 - temp_std / 10.0)  # Normalize by 10°C
            else:
                consistency_score = 0.0
                
        elif sensor_type.upper() == 'RADAR':
            # Radar consistency based on range stability
            ranges = []
            for r in readings:
                if isinstance(r, dict):
                    if 'value' in r and isinstance(r['value'], dict):
                        ranges.append(r['value'].get('range_m', 0))
                    else:
                        ranges.append(r.get('range_m', 0))
            
            if ranges:
                range_std = np.std(ranges)
                consistency_score = max(0, 1.0 - range_std / 5.0)  # Normalize by 5m
            else:
                consistency_score = 0.0
                
        else:
            # Default consistency score
            consistency_score = 0.8
        
        # Combine scores
        reliability_score = (validity_score * 0.7 + consistency_score * 0.3)
        
        return max(0.0, min(1.0, reliability_score))
        
    except Exception:
        return 0.0

def get_sensor_fusion_weights() -> Dict[str, float]:
    """
    Get recommended fusion weights for different sensor types based on reliability
    
    Returns:
        Dictionary of fusion weights
    """
    return {
        'PIR': 0.20,      # High reliability for motion detection
        'Thermal': 0.25,   # High reliability for presence detection
        'Radar': 0.20,     # Good for motion and vital signs
        'Environmental': 0.15,  # Context information
        'Audio': 0.15,     # Moderate reliability due to noise
        'Door': 0.05       # Simple binary state
    }

def estimate_sensor_deployment_cost() -> Dict[str, Dict[str, float]]:
    """
    Estimate deployment costs for sensor array
    
    Returns:
        Dictionary of cost estimates in USD
    """
    return {
        'hardware_costs': {
            'PIR_sensors': 4 * 5.0,      # 4 sensors @ $5 each
            'Thermal_camera': 1 * 45.0,   # 1 camera @ $45
            'Radar_module': 1 * 120.0,    # 1 module @ $120
            'Environmental_sensors': 3 * 15.0,  # 3 sensors @ $15 each
            'Audio_microphones': 4 * 8.0,  # 4 mics @ $8 each
            'Door_sensors': 2 * 3.0,      # 2 sensors @ $3 each
            'Processing_unit': 1 * 80.0,   # Raspberry Pi 5 4GB
            'Storage': 1 * 25.0,          # 256GB SSD
            'Power_system': 1 * 40.0,     # UPS and charging
            'Enclosure': 1 * 30.0,        # Weatherproof housing
            'Installation': 1 * 100.0     # Professional installation
        },
        'operational_costs_annual': {
            'Power_consumption': 26.2 * 24 * 365 / 1000 * 0.12,  # ~$28/year
            'Cellular_data': 12 * 20.0,    # $20/month for data
            'Maintenance': 2 * 50.0,       # 2 visits @ $50 each
            'Cloud_storage': 12 * 5.0      # $5/month for cloud backup
        },
        'total_deployment_cost': 500.0,  # Approximate total hardware cost
        'annual_operating_cost': 350.0   # Approximate annual operating cost
    }

def get_sensor_calibration_requirements() -> Dict[str, Dict[str, Any]]:
    """
    Get calibration requirements for each sensor type
    
    Returns:
        Dictionary of calibration requirements
    """
    return {
        'PIR': {
            'calibration_frequency_days': 90,
            'calibration_method': 'motion_reference',
            'drift_tolerance': 0.05,
            'auto_calibration': True
        },
        'Thermal': {
            'calibration_frequency_days': 30,
            'calibration_method': 'blackbody_reference',
            'drift_tolerance': 1.0,  # 1°C
            'auto_calibration': False
        },
        'Radar': {
            'calibration_frequency_days': 180,
            'calibration_method': 'corner_reflector',
            'drift_tolerance': 0.1,  # 10cm
            'auto_calibration': True
        },
        'Environmental': {
            'calibration_frequency_days': 365,
            'calibration_method': 'reference_standards',
            'drift_tolerance': {
                'temperature': 0.2,  # 0.2°C
                'humidity': 1.0,     # 1% RH
                'co2': 25            # 25 ppm
            },
            'auto_calibration': False
        },
        'Audio': {
            'calibration_frequency_days': 180,
            'calibration_method': 'calibrated_microphone',
            'drift_tolerance': 1.0,  # 1 dB
            'auto_calibration': True
        },
        'Door': {
            'calibration_frequency_days': 365,
            'calibration_method': 'gap_measurement',
            'drift_tolerance': 2.0,  # 2mm
            'auto_calibration': True
        }
    }
