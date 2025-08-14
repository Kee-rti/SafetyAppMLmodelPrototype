# Multi-Sensor Fusion Safety Monitoring - ML Prototype Platform

## Overview

This is a comprehensive Streamlit-based machine learning development platform for multi-sensor fusion safety monitoring systems. The application provides an end-to-end pipeline for developing, training, and deploying ML models that analyze data from multiple sensor types (PIR motion sensors, thermal cameras, radar, environmental sensors, audio arrays, and door sensors) to detect safety scenarios and assess risk levels.

The platform implements TRS-compliant dataset generation with realistic sensor fusion scenarios, supports multiple neural network architectures (LSTM, Transformer, Multi-Modal Fusion), and includes comprehensive training pipelines with experiment tracking, model evaluation, and ONNX export capabilities for edge deployment on devices like Raspberry Pi 5.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Main user interface built with Streamlit providing interactive dashboards for data visualization, model training, and evaluation
- **Plotly Integration**: Interactive visualizations for sensor data analysis, model performance metrics, and attention mechanism visualization
- **Session State Management**: Persistent state management across Streamlit sessions for experiment tracking and model persistence

### Data Management Architecture
- **TRS-Compliant Dataset Generator**: Custom dataset generation system that creates realistic multi-sensor scenarios with configurable risk levels and sensor readings
- **Sensor Data Preprocessing**: Comprehensive preprocessing pipeline supporting multiple normalization methods, outlier detection, windowing, and feature extraction
- **Feature Engineering**: Statistical and time-domain feature extraction from raw sensor data with configurable window sizes and overlap ratios

### Model Architecture
- **Multi-Modal Neural Networks**: Three main model architectures:
  - **SensorFusionLSTM**: Sequential processing with LSTM layers for temporal dependencies
  - **SensorFusionTransformer**: Attention-based architecture with multi-head attention for sensor fusion
  - **MultiModalFusionNet**: Specialized fusion network combining multiple sensor modalities
- **Fusion Techniques**: Multiple sensor fusion strategies including early fusion, feature-level fusion, late fusion, and dynamic gated fusion
- **PyTorch Backend**: All models built using PyTorch with GPU support for training acceleration

### Training and Evaluation Pipeline
- **Cross-Validation**: Stratified k-fold cross-validation for robust model evaluation
- **Hyperparameter Optimization**: Optuna integration for automated hyperparameter tuning
- **Early Stopping**: Configurable early stopping with patience and delta thresholds to prevent overfitting
- **Comprehensive Metrics**: Full evaluation suite including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices

### Model Interpretability
- **SHAP Integration**: SHAP (SHapley Additive exPlanations) values for feature importance analysis
- **Attention Visualization**: Visualization of attention weights in transformer models to understand sensor importance
- **Feature Importance Analysis**: Sensor group analysis and temporal importance visualization

### Deployment Architecture
- **ONNX Export**: Model conversion to ONNX format for cross-platform deployment
- **Edge Optimization**: Specific optimizations for Raspberry Pi 5 and AI HAT+ deployment
- **Quantization Support**: INT8 quantization options for reduced model size and faster inference
- **Performance Benchmarking**: Inference time and memory usage analysis for edge deployment validation

### Experiment Management
- **MLflow Integration**: Experiment tracking with MLflow for model versioning, parameter logging, and metric tracking
- **Session Persistence**: Local session state management for maintaining experiment history within Streamlit sessions
- **Model Serialization**: Pickle-based model saving and loading for experiment reproducibility

## External Dependencies

### Machine Learning Libraries
- **PyTorch**: Core deep learning framework for model development and training
- **Scikit-learn**: Traditional ML algorithms, preprocessing utilities, and evaluation metrics
- **NumPy/Pandas**: Data manipulation and numerical computing
- **ONNX/ONNXRuntime**: Model serialization and cross-platform inference

### Hyperparameter Optimization
- **Optuna**: Automated hyperparameter optimization with pruning and efficient search algorithms

### Experiment Tracking
- **MLflow**: Experiment tracking, model registry, and lifecycle management

### Visualization and UI
- **Streamlit**: Web application framework for the main user interface
- **Plotly**: Interactive plotting library for dynamic visualizations
- **Matplotlib/Seaborn**: Statistical plotting and heatmap visualizations

### Model Interpretability
- **SHAP**: Model explainability and feature importance analysis (optional dependency)

### Data Processing
- **SciPy**: Scientific computing functions for signal processing and statistical analysis

### Sensor Hardware Integration
The platform is designed to work with specific sensor hardware:
- **PIR Motion Sensors**: HC-SR501 for motion detection
- **Thermal Cameras**: AMG8833 for presence detection with 8x8 thermal grids
- **60GHz Radar**: BGT60TR13C for vital signs and motion analysis
- **Environmental Sensors**: DS18B20+, SHT40, SCD40 for temperature, humidity, and CO2 monitoring
- **Audio Arrays**: INMP441 4-microphone arrays for spatial audio processing
- **Door Sensors**: MC-38 magnetic and A3144 Hall effect sensors for access monitoring

### Target Deployment Platform
- **Raspberry Pi 5**: Primary edge computing platform
- **AI HAT+**: Hardware acceleration accessory for optimized inference performance