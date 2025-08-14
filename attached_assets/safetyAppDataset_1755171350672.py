#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


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

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ScenarioType(Enum):
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


class TRSCompliantDatasetGenerator:
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.specs = {
            "pir": {
                "model": "HC-SR501",
                "count": 4,
                "detection_range_m": (3, 7),
                "detection_angle_deg": 120,
                "operating_voltage": (4.5, 20.0),
                "operating_current_ua": 50,
                "output_voltage": {"high": 3.3, "low": 0.0},
                "delay_time_s": (0.3, 18),
                "block_time_s": 2.5,
                "operating_temp_c": (-15, 70),
                "false_positive_rate": 0.02,
                "sensitivity": 0.8,
                "power_consumption_mw": 0.165
            },
            "thermal": {
                "model": "AMG8833",
                "resolution": (8, 8),
                "field_of_view_deg": (60, 60),
                "temp_range_c": (0, 80),
                "accuracy_c": 2.5,
                "refresh_rate_hz": 10,
                "interface": "I2C",
                "operating_voltage": (3.3, 5.0),
                "operating_current_ma": 4.5,
                "power_consumption_mw": 14.85
            },
            "radar": {
                "model": "BGT60TR13C",
                "frequency_range_ghz": (57, 64),
                "rf_output_power_dbm": 12,
                "detection_range_m": (0.15, 10),
                "range_resolution_cm": 15,
                "velocity_range_mps": (-8.1, 8.1),
                "velocity_resolution_mps": 0.2,
                "update_rate_hz": 100,
                "interface": "SPI",
                "operating_voltage": 3.3,
                "operating_current_ma": 85,
                "power_consumption_mw": 280.5
            },
            "environment": {
                "temperature": {
                    "model": "DS18B20+",
                    "range_c": (-55, 125),
                    "accuracy_c": 0.5,
                    "resolution_bits": 12,
                    "interface": "1-Wire",
                    "operating_voltage": (3.0, 5.5),
                    "power_consumption_mw": 4.5
                },
                "humidity": {
                    "model": "SHT40-AD1B", 
                    "temp_range_c": (-40, 125),
                    "temp_accuracy_c": 0.2,
                    "humidity_range_pct": (0, 100),
                    "humidity_accuracy_pct": 1.8,
                    "response_time_s": 1,
                    "interface": "I2C",
                    "operating_voltage": (1.08, 3.6),
                    "power_consumption_mw": 1.0
                },
                "co2": {
                    "model": "SCD40-D-R2",
                    "range_ppm": (0, 40000),
                    "accuracy": "±(50 ppm + 5% of reading)",
                    "repeatability_ppm": 10,
                    "response_time_s": 60,
                    "interface": "I2C", 
                    "operating_voltage": (2.4, 5.5),
                    "operating_current_ma": 18,
                    "power_consumption_mw": 59.4
                }
            },
            "audio": {
                "model": "INMP441",
                "count": 4,
                "acoustic_overload_point_db": 120,
                "snr_db": 61,
                "sensitivity_dbfs": -26,
                "frequency_response_hz": (60, 15000),
                "interface": "I2S",
                "operating_voltage": (1.8, 3.3),
                "operating_current_ma": 1.4,
                "power_consumption_mw": 4.62
            },
            "door": {
                "magnetic": {
                    "model": "MC-38",
                    "contact_type": "NO",
                    "contact_rating": {"voltage": 200, "current": 0.5},
                    "gap_distance_mm": 20,
                    "operating_temp_c": (-10, 55),
                    "power_consumption_mw": 0.0
                },
                "hall_effect": {
                    "model": "A3144EUA-T",
                    "operating_voltage": (4.5, 24),
                    "output_type": "open_collector",
                    "magnetic_threshold_gauss": {"operate": 280, "release": 200},
                    "operating_temp_c": (-40, 85),
                    "power_consumption_mw": 15.0
                }
            }
        }
        
        self.power_budget = {
            "raspberry_pi_5": {"voltage": 5.0, "current_ma": 3000, "power_w": 15.0},
            "nvme_ssd": {"voltage": 5.0, "current_ma": 500, "power_w": 2.5},
            "gsm_module": {"voltage": 3.3, "current_ma": 2000, "power_w": 6.6},
            "total_sensors_w": 0.38,
            "total_system_w": 26.2,
            "battery_wh": 57.72,
            "runtime_hours": 24
        }
        
        self.environmental_limits = {
            "operating_temp_c": (-20, 60),
            "storage_temp_c": (-40, 85), 
            "humidity_pct": (5, 95),
            "altitude_m": 3000,
            "vibration_g": 10,
            "ip_rating": "IP65"
        }
        
        self.communication = {
            "wifi": {"standard": "802.11ac", "bands": ["2.4GHz", "5.0GHz"]},
            "bluetooth": {"version": "5.0", "ble": True},
            "gsm": {
                "model": "SIM7600G-H", 
                "standards": ["LTE-FDD", "LTE-TDD", "WCDMA", "GSM/GPRS/EDGE"],
                "data_speed": {"dl_mbps": 150, "ul_mbps": 50}
            },
            "ethernet": {"speed": "Gigabit", "type": "BCM54213PE"}
        }
        
        self.scenarios = {
            ScenarioType.NORMAL_ACTIVITY: {
                "duration_h": (1, 8), "move_freq": 0.7, "risk": RiskLevel.LOW,
                "desc": "Regular daily activities with normal movement patterns",
                "detection_accuracy_target": 0.998
            },
            ScenarioType.NO_MOVEMENT: {
                "duration_h": (2, 12), "move_freq": 0.05, "risk": RiskLevel.MEDIUM,
                "desc": "Extended period without movement - possible sleep/rest",
                "detection_accuracy_target": 0.998  # <-- changed from 0.995
            },
            ScenarioType.PROLONGED_SILENCE: {
                "duration_h": (1, 4), "move_freq": 0.1, "risk": RiskLevel.HIGH,
                "desc": "Extended silence with minimal audio and motion",
                "detection_accuracy_target": 0.998
            },
            ScenarioType.FALL_DETECTED: {
                "duration_h": (0.1, 2), "move_freq": 0.0, "risk": RiskLevel.CRITICAL,
                "desc": "Fall signature followed by stillness",
                "detection_accuracy_target": 0.999
            },
            ScenarioType.MEDICAL_EMERGENCY: {
                "duration_h": (0.5, 6), "move_freq": 0.2, "risk": RiskLevel.CRITICAL,
                "desc": "Medical distress with abnormal vital signs",
                "detection_accuracy_target": 0.999
            },
            ScenarioType.DOOR_LEFT_OPEN: {
                "duration_h": (0.5, 24), "move_freq": 0.3, "risk": RiskLevel.MEDIUM,
                "desc": "Door left open for extended period",
                "detection_accuracy_target": 0.998  # <-- changed from 0.995
            },
            ScenarioType.WANDERING_RISK: {
                "duration_h": (0.2, 2), "move_freq": 0.6, "risk": RiskLevel.HIGH,
                "desc": "Door opened during quiet hours without caregiver",
                "detection_accuracy_target": 0.998
            },
            ScenarioType.ENTRAPMENT_RISK: {
                "duration_h": (0.5, 6), "move_freq": 0.05, "risk": RiskLevel.HIGH,
                "desc": "Person trapped in room/bathroom",
                "detection_accuracy_target": 0.998
            },
            ScenarioType.INTRUSION_ALERT: {
                "duration_h": (0.1, 1), "move_freq": 0.6, "risk": RiskLevel.CRITICAL,
                "desc": "Unauthorized entry while security system active",
                "detection_accuracy_target": 0.999
            },
            ScenarioType.ENVIRONMENTAL_HAZARD: {
                "duration_h": (0.1, 12), "move_freq": 0.4, "risk": RiskLevel.HIGH,
                "desc": "Environmental parameters beyond safe thresholds",
                "detection_accuracy_target": 0.998
            }
        }
        
        self._last_detection = {}

    def validate_against_trs(self) -> Dict[str, Any]:
        validation_report = {
            "compliant": True,
            "issues": [],
            "recommendations": [],
            "power_analysis": {},
            "sensor_validation": {}
        }
        
        total_sensor_power = 0
        for sensor_type, specs in self.specs.items():
            if sensor_type == "pir":
                power = specs["power_consumption_mw"] * specs["count"]
                total_sensor_power += power
            elif sensor_type == "thermal":
                total_sensor_power += specs["power_consumption_mw"]
            elif sensor_type == "radar":
                total_sensor_power += specs["power_consumption_mw"]
            elif sensor_type == "environment":
                for env_sensor in specs.values():
                    if isinstance(env_sensor, dict) and "power_consumption_mw" in env_sensor:
                        total_sensor_power += env_sensor["power_consumption_mw"]
            elif sensor_type == "audio":
                power = specs["power_consumption_mw"] * specs["count"] 
                total_sensor_power += power
            elif sensor_type == "door":
                for door_sensor in specs.values():
                    if isinstance(door_sensor, dict) and "power_consumption_mw" in door_sensor:
                        total_sensor_power += door_sensor["power_consumption_mw"]
        
        validation_report["power_analysis"] = {
            "total_sensor_power_mw": total_sensor_power,
            "total_sensor_power_w": total_sensor_power / 1000,
            "within_budget": total_sensor_power / 1000 < self.power_budget["total_sensors_w"],
            "power_efficiency": (total_sensor_power / 1000) / self.power_budget["total_sensors_w"]
        }
        
        critical_sensors = ["pir", "thermal", "radar", "environment"] 
        for sensor in critical_sensors:
            if sensor not in self.specs:
                validation_report["issues"].append(f"Missing critical sensor: {sensor}")
                validation_report["compliant"] = False
        
        for scenario, config in self.scenarios.items():
            if config["detection_accuracy_target"] < 0.998:
                validation_report["issues"].append(
                    f"Scenario {scenario.value} accuracy target {config['detection_accuracy_target']} below TRS requirement (≥99.8%)"
                )
        
        required_comm = ["wifi", "bluetooth", "gsm", "ethernet"]
        for comm in required_comm:
            if comm not in self.communication:
                validation_report["issues"].append(f"Missing communication interface: {comm}")
        
        return validation_report

    def generate_trs_compliant_pir(self, scenario: ScenarioType, duration_h: float, t0: float) -> List[SensorReading]:
        readings = []
        steps = int(duration_h * 3600 // 60)  # One reading per minute
        base_freq = self.scenarios[scenario]["move_freq"]
        
        for sid in range(self.specs["pir"]["count"]):
            sensor_id = f"PIR_{sid:02d}"
            
            sensor_health = np.random.uniform(0.95, 1.0)
            calibration_drift = np.random.uniform(-0.05, 0.05)
            
            for i in range(steps):
                ts = t0 + i
                
                p = base_freq
                
                if scenario in [ScenarioType.NO_MOVEMENT, ScenarioType.PROLONGED_SILENCE, ScenarioType.ENTRAPMENT_RISK]:
                    p = 0.01
                elif scenario == ScenarioType.FALL_DETECTED:
                    p = 1.0 if i < 3 else 0.005
                elif scenario == ScenarioType.WANDERING_RISK:
                    p = 0.8 if i % 120 < 10 else 0.1
                
                detect = np.random.rand() < p
                
                if not detect and np.random.rand() < self.specs["pir"]["false_positive_rate"]:
                    detect = True
                
                if detect and np.random.rand() < (1 - self.specs["pir"]["sensitivity"]):
                    detect = False
                
                if detect and np.random.rand() < (1 - sensor_health):
                    detect = False
                
                if hasattr(self, '_last_detection'):
                    if self._last_detection.get(sensor_id, 0) + self.specs["pir"]["block_time_s"] > (ts - t0):
                        detect = False
                else:
                    self._last_detection = {}
                
                if detect:
                    self._last_detection[sensor_id] = ts - t0
                
                val = self.specs["pir"]["output_voltage"]["high"] if detect else self.specs["pir"]["output_voltage"]["low"]
                confidence = 0.9 if detect else 0.95
                
                detection_range = np.random.uniform(*self.specs["pir"]["detection_range_m"])
                
                readings.append(SensorReading(
                    sensor_id=sensor_id,
                    sensor_type="pir", 
                    value=val,
                    confidence=confidence * sensor_health,
                    timestamp=ts,
                    metadata={
                        "detection_range_m": float(detection_range),
                        "zone": sid + 1,
                        "model": self.specs["pir"]["model"],
                        "block_time_s": self.specs["pir"]["block_time_s"]
                    },
                    power_consumption_mw=self.specs["pir"]["power_consumption_mw"],
                    sensor_health=sensor_health,
                    calibration_drift=calibration_drift
                ))
        
        return readings

    def generate_trs_compliant_thermal(self, scenario: ScenarioType, duration_h: float, t0: float) -> List[SensorReading]:
        readings = []
        res = self.specs["thermal"]["resolution"]
        refresh_rate = self.specs["thermal"]["refresh_rate_hz"]
        steps = int(duration_h * 3600 // 60)  # One reading per minute (was: int(duration_h * 3600 * refresh_rate / 10))
        
        base_room_temp = np.random.uniform(20, 25)
        sensor_health = np.random.uniform(0.95, 1.0)
        calibration_drift = np.random.uniform(-0.1, 0.1)
        
        for i in range(steps):
            ts = t0 + i * 60  # 1 minute intervals
            
            matrix = np.full(res, base_room_temp, dtype=float)
            
            present = np.random.rand() < self.scenarios[scenario]["move_freq"]
            if present:
                human_temp = np.random.uniform(35.5, 37.5)
                
                px, py = np.random.randint(1, res[0]-2), np.random.randint(1, res[1]-2)
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= px+dx < res[0] and 0 <= py+dy < res[1]:
                            intensity = 1.0 if (dx == 0 and dy == 0) else 0.6
                            matrix[px+dx, py+dy] = human_temp * intensity + base_room_temp * (1-intensity)
            
            if scenario == ScenarioType.FALL_DETECTED and i < 5:
                matrix[3:5, 1:7] = np.random.uniform(34, 36, (2, 6))
                
            elif scenario == ScenarioType.MEDICAL_EMERGENCY:
                if present:
                    fever_temp = np.random.uniform(38, 40)
                    matrix[px-1:px+2, py-1:py+2] = fever_temp + np.random.normal(0, 0.5, (3,3))
                    
            elif scenario == ScenarioType.ENVIRONMENTAL_HAZARD:
                matrix += np.random.uniform(10, 20)
            
            noise_std = self.specs["thermal"]["accuracy_c"] / 3
            matrix += np.random.normal(0, noise_std, size=res)
            
            matrix += calibration_drift
            
            matrix = np.clip(matrix, *self.specs["thermal"]["temp_range_c"])
            
            t_max = float(matrix.max())
            t_avg = float(matrix.mean())
            t_var = float(matrix.var())
            t_min = float(matrix.min())
            
            temp_diff = t_max - base_room_temp
            confidence = min(0.9, 0.5 + temp_diff / 10) * sensor_health
            
            readings.append(SensorReading(
                sensor_id="THERMAL_01",
                sensor_type="thermal",
                value=t_max,
                confidence=confidence,
                timestamp=ts,
                metadata={
                    # "matrix_8x8": matrix.tolist(),  # Already commented out for optimization
                    "avg_temp_c": t_avg,
                    "var_temp": t_var,
                    "min_temp_c": t_min,
                    "model": self.specs["thermal"]["model"],
                    "resolution": f"{res[0]}x{res[1]}",
                    "field_of_view_deg": self.specs["thermal"]["field_of_view_deg"]
                },
                power_consumption_mw=self.specs["thermal"]["power_consumption_mw"],
                sensor_health=sensor_health,
                calibration_drift=calibration_drift
            ))
        
        return readings

    def generate_trs_compliant_radar(self, scenario: ScenarioType, duration_h: float, t0: float) -> List[SensorReading]:
        readings = []
        update_rate = min(self.specs["radar"]["update_rate_hz"], 10)
        steps = int(duration_h * 3600 // 60)  # One reading per minute (was: int(duration_h * 3600 * update_rate / 10))
        
        sensor_health = np.random.uniform(0.95, 1.0)
        calibration_drift = np.random.uniform(-0.02, 0.02)
        
        for i in range(steps):
            ts = t0 + i * 60  # 1 minute intervals
            
            distance = float('inf')
            velocity = 0.0
            confidence = 0.5
            
            present = np.random.rand() < self.scenarios[scenario]["move_freq"]
            
            if present:
                distance = np.random.uniform(0.2, 8.0)
                
                if scenario == ScenarioType.NORMAL_ACTIVITY:
                    velocity = np.random.uniform(-2.0, 2.0)
                elif scenario == ScenarioType.FALL_DETECTED and i < 5:
                    velocity = np.random.uniform(-3.5, -1.0)
                elif scenario in [ScenarioType.NO_MOVEMENT, ScenarioType.PROLONGED_SILENCE]:
                    velocity = np.random.uniform(-0.1, 0.1)
                elif scenario == ScenarioType.WANDERING_RISK:
                    velocity = np.random.uniform(-1.5, 1.5)
                else:
                    velocity = np.random.uniform(-2.5, 2.5)
                
                distance = round(distance / self.specs["radar"]["range_resolution_cm"] * 100) * self.specs["radar"]["range_resolution_cm"] / 100
                velocity = round(velocity / self.specs["radar"]["velocity_resolution_mps"]) * self.specs["radar"]["velocity_resolution_mps"]
                
                confidence = np.random.uniform(0.85, 0.95)
            
            if distance != float('inf'):
                distance += calibration_drift + np.random.normal(0, 0.05)
                velocity += np.random.normal(0, 0.05)
                
                distance = max(self.specs["radar"]["detection_range_m"][0], 
                             min(distance, self.specs["radar"]["detection_range_m"][1]))
                velocity = max(self.specs["radar"]["velocity_range_mps"][0],
                             min(velocity, self.specs["radar"]["velocity_range_mps"][1]))
            
            value = {
                "distance_m": 999.0 if distance == float('inf') else round(distance, 3),
                "velocity_mps": round(velocity, 3),
                "range_bins": int(distance / self.specs["radar"]["range_resolution_cm"] * 100) if distance != float('inf') else 0
            }
            
            readings.append(SensorReading(
                sensor_id="RADAR_01",
                sensor_type="radar",
                value=value,
                confidence=confidence * sensor_health,
                timestamp=ts,
                metadata={
                    "model": self.specs["radar"]["model"],
                    "frequency_ghz": np.random.uniform(*self.specs["radar"]["frequency_range_ghz"]),
                    "rf_power_dbm": self.specs["radar"]["rf_output_power_dbm"],
                    "range_resolution_cm": self.specs["radar"]["range_resolution_cm"],
                    "velocity_resolution_mps": self.specs["radar"]["velocity_resolution_mps"]
                },
                power_consumption_mw=self.specs["radar"]["power_consumption_mw"],
                sensor_health=sensor_health,
                calibration_drift=calibration_drift
            ))
        
        return readings

    def generate_trs_compliant_audio(self, scenario: ScenarioType, duration_h: float, t0: float) -> List[SensorReading]:
        readings = []
        steps = int(duration_h * 3600 // 60)  # One reading per minute (was: int(duration_h * 3600 / 10))
        
        for mic_id in range(self.specs["audio"]["count"]):
            sensor_id = f"MIC_{mic_id:02d}"
            sensor_health = np.random.uniform(0.95, 1.0)
            
            for i in range(steps):
                ts = t0 + i * 60  # 1 minute intervals
                
                base_noise = np.random.uniform(30, 40)
                
                sound_level = base_noise
                sound_freq_hz = 0
                confidence = 0.6
                
                if scenario == ScenarioType.NORMAL_ACTIVITY:
                    if np.random.rand() < 0.3:
                        sound_level = np.random.uniform(50, 70)
                        sound_freq_hz = np.random.uniform(200, 2000)
                        confidence = 0.8
                elif scenario == ScenarioType.PROLONGED_SILENCE:
                    sound_level = np.random.uniform(28, 35)
                    confidence = 0.9
                elif scenario == ScenarioType.FALL_DETECTED:
                    if i < 2:
                        sound_level = np.random.uniform(80, 100)
                        sound_freq_hz = np.random.uniform(100, 500)
                        confidence = 0.95
                    elif 2 <= i < 5:
                        sound_level = np.random.uniform(60, 80)
                        sound_freq_hz = np.random.uniform(50, 300)
                        confidence = 0.9
                elif scenario == ScenarioType.MEDICAL_EMERGENCY:
                    if np.random.rand() < 0.4:
                        sound_level = np.random.uniform(65, 95)
                        sound_freq_hz = np.random.uniform(300, 1500)
                        confidence = 0.9
                elif scenario == ScenarioType.WANDERING_RISK:
                    if np.random.rand() < 0.2:
                        sound_level = np.random.uniform(45, 65)
                        sound_freq_hz = np.random.uniform(100, 800)
                        confidence = 0.8
                elif scenario == ScenarioType.INTRUSION_ALERT:
                    if np.random.rand() < 0.6:
                        sound_level = np.random.uniform(55, 85)
                        sound_freq_hz = np.random.uniform(200, 1200)
                        confidence = 0.85
                
                sound_level = np.clip(sound_level, 0, self.specs["audio"]["acoustic_overload_point_db"])
                sound_level += np.random.normal(0, 2)
                
                readings.append(SensorReading(
                    sensor_id=sensor_id,
                    sensor_type="audio",
                    value={
                        "sound_level_db": round(sound_level, 1),
                        "dominant_freq_hz": round(sound_freq_hz, 1),
                        "snr_db": self.specs["audio"]["snr_db"]
                    },
                    confidence=confidence * sensor_health,
                    timestamp=ts,
                    metadata={
                        "model": self.specs["audio"]["model"],
                        "sensitivity_dbfs": self.specs["audio"]["sensitivity_dbfs"],
                        "frequency_range_hz": self.specs["audio"]["frequency_response_hz"],
                        "microphone_position": mic_id + 1
                    },
                    power_consumption_mw=self.specs["audio"]["power_consumption_mw"],
                    sensor_health=sensor_health,
                    calibration_drift=np.random.uniform(-1, 1)
                ))
        
        return readings

    def generate_trs_compliant_door(self, scenario: ScenarioType, duration_h: float, t0: float) -> List[SensorReading]:
        readings = []
        steps = int(duration_h * 3600 // 60)  # One reading per minute (was: int(duration_h * 3600))
        
        for door_id in range(2):
            mag_sensor_id = f"MAG_{door_id:02d}"
            hall_sensor_id = f"HALL_{door_id:02d}"
            
            mag_health = np.random.uniform(0.98, 1.0)
            hall_health = np.random.uniform(0.95, 1.0)
            
            door_open = False
            last_state_change = 0
            
            for i in range(steps):
                ts = t0 + i * 60  # 1 minute intervals
                
                if scenario == ScenarioType.DOOR_LEFT_OPEN:
                    if i < 1:  # Only first minute
                        door_open = True
                    elif i == 1:
                        door_open = False if np.random.rand() < 0.3 else True
                elif scenario == ScenarioType.WANDERING_RISK:
                    hour = ((ts % 86400) / 3600.0) % 24
                    if 22 <= hour or hour <= 6:
                        if np.random.rand() < 0.001:
                            door_open = not door_open
                            last_state_change = i
                elif scenario == ScenarioType.ENTRAPMENT_RISK:
                    if i < 1:
                        door_open = False
                    elif i == 1 and door_id == 0:
                        door_open = True
                elif scenario == ScenarioType.INTRUSION_ALERT:
                    if np.random.rand() < 0.002:
                        door_open = not door_open
                        last_state_change = i
                else:
                    if np.random.rand() < 0.0002:
                        door_open = not door_open
                        last_state_change = i
                
                mag_value = 0 if door_open else 1
                hall_value = 1 if door_open else 0
                
                if np.random.rand() < 0.001:
                    mag_value = 1 - mag_value
                if np.random.rand() < 0.002:
                    hall_value = 1 - hall_value
                
                mag_confidence = 0.98 * mag_health
                hall_confidence = 0.95 * hall_health
                
                mag_reading = SensorReading(
                    sensor_id=mag_sensor_id,
                    sensor_type="magnetic",
                    value=mag_value,
                    confidence=mag_confidence,
                    timestamp=ts,
                    metadata={
                        "model": self.specs["door"]["magnetic"]["model"],
                        "contact_type": self.specs["door"]["magnetic"]["contact_type"],
                        "gap_distance_mm": self.specs["door"]["magnetic"]["gap_distance_mm"],
                        "door_position": door_id + 1,
                        "state_duration_s": (i - last_state_change) * 60
                    },
                    power_consumption_mw=self.specs["door"]["magnetic"]["power_consumption_mw"],
                    sensor_health=mag_health,
                    calibration_drift=0.0
                )
                
                hall_reading = SensorReading(
                    sensor_id=hall_sensor_id,
                    sensor_type="hall_effect",
                    value=hall_value,
                    confidence=hall_confidence,
                    timestamp=ts,
                    metadata={
                        "model": self.specs["door"]["hall_effect"]["model"],
                        "magnetic_threshold_gauss": self.specs["door"]["hall_effect"]["magnetic_threshold_gauss"],
                        "door_position": door_id + 1,
                        "state_duration_s": (i - last_state_change) * 60
                    },
                    power_consumption_mw=self.specs["door"]["hall_effect"]["power_consumption_mw"],
                    sensor_health=hall_health,
                    calibration_drift=0.0
                )
                
                readings.extend([mag_reading, hall_reading])
        
        return readings

    def generate_trs_compliant_environment(self, scenario: ScenarioType, duration_h: float, t0: float) -> List[SensorReading]:
        readings = []
        steps = int(duration_h * 3600 // 60)  # One reading per minute (was: int(duration_h * 3600 / 60))
        
        base_temp = np.random.uniform(20, 25)
        base_humidity = np.random.uniform(40, 60) 
        base_co2 = np.random.uniform(400, 600)
        
        temp_health = np.random.uniform(0.95, 1.0)
        humidity_health = np.random.uniform(0.95, 1.0)
        co2_health = np.random.uniform(0.90, 1.0)
        
        for i in range(steps):
            ts = t0 + i * 60  # 1 minute intervals
            hour = ((ts % 86400) / 3600.0)
            
            temp_variation = 3 * np.sin(2 * np.pi * (hour - 6) / 24)
            humidity_variation = -10 * np.sin(2 * np.pi * (hour - 12) / 24)
            
            temperature = base_temp + temp_variation
            humidity = base_humidity + humidity_variation  
            co2 = base_co2
            
            if scenario == ScenarioType.ENVIRONMENTAL_HAZARD:
                temperature += np.random.uniform(15, 25)
                humidity += np.random.uniform(20, 35)
                co2 += np.random.uniform(2000, 4000)
            elif scenario in [ScenarioType.ENTRAPMENT_RISK, ScenarioType.NO_MOVEMENT]:
                co2 += np.random.uniform(200, 800) * (i / steps)
                
            if self.scenarios[scenario]["move_freq"] > 0.5:
                co2 += np.random.uniform(100, 300)
                temperature += np.random.uniform(1, 3)
                humidity += np.random.uniform(-5, 10)
            
            temperature += np.random.normal(0, self.specs["environment"]["temperature"]["accuracy_c"])
            humidity += np.random.normal(0, self.specs["environment"]["humidity"]["humidity_accuracy_pct"])
            
            co2_accuracy = 50 + 0.05 * co2
            co2 += np.random.normal(0, co2_accuracy)
            
            temperature = np.clip(temperature, *self.specs["environment"]["temperature"]["range_c"])
            humidity = np.clip(humidity, 0, 100)
            co2 = np.clip(co2, 0, self.specs["environment"]["co2"]["range_ppm"][1])
            
            temp_confidence = 0.95 * temp_health
            humidity_confidence = 0.93 * humidity_health
            co2_confidence = 0.90 * co2_health
            
            temp_reading = SensorReading(
                sensor_id="TEMP_01",
                sensor_type="temperature",
                value=round(temperature, 2),
                confidence=temp_confidence,
                timestamp=ts,
                metadata={
                    "model": self.specs["environment"]["temperature"]["model"],
                    "resolution_bits": self.specs["environment"]["temperature"]["resolution_bits"],
                    "interface": self.specs["environment"]["temperature"]["interface"]
                },
                power_consumption_mw=self.specs["environment"]["temperature"]["power_consumption_mw"],
                sensor_health=temp_health,
                calibration_drift=np.random.uniform(-0.1, 0.1)
            )
            
            humidity_reading = SensorReading(
                sensor_id="HUMIDITY_01",
                sensor_type="humidity",
                value=round(humidity, 1),
                confidence=humidity_confidence,
                timestamp=ts,
                metadata={
                    "model": self.specs["environment"]["humidity"]["model"],
                    "response_time_s": self.specs["environment"]["humidity"]["response_time_s"],
                    "interface": self.specs["environment"]["humidity"]["interface"]
                },
                power_consumption_mw=self.specs["environment"]["humidity"]["power_consumption_mw"],
                sensor_health=humidity_health,
                calibration_drift=np.random.uniform(-2, 2)
            )
            
            co2_reading = SensorReading(
                sensor_id="CO2_01",
                sensor_type="co2",
                value=round(co2, 0),
                confidence=co2_confidence,
                timestamp=ts,
                metadata={
                    "model": self.specs["environment"]["co2"]["model"],
                    "range_ppm": self.specs["environment"]["co2"]["range_ppm"],
                    "response_time_s": self.specs["environment"]["co2"]["response_time_s"]
                },
                power_consumption_mw=self.specs["environment"]["co2"]["power_consumption_mw"],
                sensor_health=co2_health,
                calibration_drift=np.random.uniform(-20, 20)
            )
            
            readings.extend([temp_reading, humidity_reading, co2_reading])
        
        return readings

    def generate_scenario_dataset(self, scenario: ScenarioType, num_instances: int = 10) -> List[Dict[str, Any]]:
        dataset = []
        
        for instance in range(num_instances):
            scenario_config = self.scenarios[scenario]
            duration = np.random.uniform(*scenario_config["duration_h"])
            start_time = np.random.uniform(0, 86400)
            
            pir_data = self.generate_trs_compliant_pir(scenario, duration, start_time)
            thermal_data = self.generate_trs_compliant_thermal(scenario, duration, start_time)
            radar_data = self.generate_trs_compliant_radar(scenario, duration, start_time)
            env_data = self.generate_trs_compliant_environment(scenario, duration, start_time)
            audio_data = self.generate_trs_compliant_audio(scenario, duration, start_time)
            door_data = self.generate_trs_compliant_door(scenario, duration, start_time)
            
            all_readings = pir_data + thermal_data + radar_data + env_data + audio_data + door_data
            all_readings.sort(key=lambda x: x.timestamp)
            
            total_power = sum(r.power_consumption_mw for r in all_readings) / len(all_readings)
            avg_confidence = sum(r.confidence for r in all_readings) / len(all_readings)
            
            dataset.append({
               "scenario": scenario.value,
               "instance_id": instance,
               "duration_hours": duration,
               "start_timestamp": start_time,
               "risk_level": scenario_config["risk"].value,
               "description": scenario_config["desc"],
               "detection_target": scenario_config["detection_accuracy_target"],
               "sensor_readings": [asdict(r) for r in all_readings],
               "summary_stats": {
                   "total_readings": len(all_readings),
                   "avg_power_consumption_mw": round(total_power, 3),
                   "avg_confidence": round(avg_confidence, 3),
                   "pir_activations": sum(1 for r in pir_data if r.value > 0),
                   "thermal_detections": sum(1 for r in thermal_data if r.value > 30),
                   "radar_detections": sum(1 for r in radar_data if r.value["distance_m"] < 900),
                   "audio_events": sum(1 for r in audio_data if r.value["sound_level_db"] > 45),
                   "door_state_changes": sum(1 for r in door_data if r.sensor_type == "magnetic")
               }
           })
        
        return dataset

    def generate_full_trs_dataset(self, instances_per_scenario: int = 50) -> Dict[str, Any]:
        full_dataset = {
            "metadata": {
                "generator_version": "1.0.0",
                "generation_timestamp": datetime.now().isoformat(),
                "trs_compliance_check": self.validate_against_trs(),
                "total_scenarios": len(ScenarioType),
                "instances_per_scenario": instances_per_scenario,
                "sensor_specifications": self.specs,
                "power_budget": self.power_budget,
                "environmental_limits": self.environmental_limits,
                "communication_specs": self.communication
            },
            "scenarios": {},
            "aggregated_stats": {}
        }
        
        all_instances = []
        scenario_stats = {}
        
        for scenario in ScenarioType:
            print(f"Generating {instances_per_scenario} instances for {scenario.value}...")
            scenario_data = self.generate_scenario_dataset(scenario, instances_per_scenario)
            full_dataset["scenarios"][scenario.value] = scenario_data
            all_instances.extend(scenario_data)
            
            scenario_readings = []
            for instance in scenario_data:
                scenario_readings.extend(instance["sensor_readings"])
            
            scenario_stats[scenario.value] = {
                "total_instances": len(scenario_data),
                "total_readings": len(scenario_readings),
                "avg_duration_hours": np.mean([inst["duration_hours"] for inst in scenario_data]),
                "avg_confidence": np.mean([r["confidence"] for r in scenario_readings]),
                "avg_power_mw": np.mean([r["power_consumption_mw"] for r in scenario_readings]),
                "sensor_type_distribution": {}
            }
            
            sensor_types = [r["sensor_type"] for r in scenario_readings]
            for sensor_type in set(sensor_types):
                scenario_stats[scenario.value]["sensor_type_distribution"][sensor_type] = sensor_types.count(sensor_type)
        
        all_readings = []
        for instance in all_instances:
            all_readings.extend(instance["sensor_readings"])
        
        full_dataset["aggregated_stats"] = {
            "total_instances": len(all_instances),
            "total_sensor_readings": len(all_readings),
            "avg_readings_per_instance": len(all_readings) / len(all_instances),
            "total_power_consumption_mw": sum(r["power_consumption_mw"] for r in all_readings),
            "avg_sensor_health": np.mean([r["sensor_health"] for r in all_readings]),
            "confidence_distribution": {
                "mean": np.mean([r["confidence"] for r in all_readings]),
                "std": np.std([r["confidence"] for r in all_readings]),
                "min": np.min([r["confidence"] for r in all_readings]),
                "max": np.max([r["confidence"] for r in all_readings])
            },
            "scenario_stats": scenario_stats,
            "dataset_size_mb": len(json.dumps(full_dataset, default=str)) / (1024 * 1024)
        }
        
        return full_dataset

    def export_dataset(self, dataset: Dict[str, Any], output_path: str = "trs_compliant_dataset.json"):
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        print(f"Dataset exported to {output_path}")
        
        summary_df = pd.DataFrame([
            {
                "scenario": scenario,
                "instances": len(data),
                "avg_duration_h": np.mean([inst["duration_hours"] for inst in data]),
                "risk_level": data[0]["risk_level"],
                "total_readings": sum(inst["summary_stats"]["total_readings"] for inst in data)
            }
            for scenario, data in dataset["scenarios"].items()
        ])
        
        summary_path = output_path.replace(".json", "_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary exported to {summary_path}")

if __name__ == "__main__":
    generator = TRSCompliantDatasetGenerator(seed=42)
    
    validation = generator.validate_against_trs()
    print("TRS Compliance Check:")
    print(f"Compliant: {validation['compliant']}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    print(f"\nPower Analysis:")
    print(f"Total sensor power: {validation['power_analysis']['total_sensor_power_w']:.3f}W")
    print(f"Power efficiency: {validation['power_analysis']['power_efficiency']:.1%}")
    
    dataset = generator.generate_full_trs_dataset(instances_per_scenario=5)  # Reduced from 25
    
    print(f"\nDataset Generation Complete:")
    print(f"Total instances: {dataset['aggregated_stats']['total_instances']}")
    print(f"Total readings: {dataset['aggregated_stats']['total_sensor_readings']:,}")
    print(f"Dataset size: {dataset['aggregated_stats']['dataset_size_mb']:.1f} MB")
    print(f"Average confidence: {dataset['aggregated_stats']['confidence_distribution']['mean']:.3f}")
    
    generator.export_dataset(dataset, "ai_safety_monitor_dataset.json")