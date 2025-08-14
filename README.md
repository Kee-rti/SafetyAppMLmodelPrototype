# Multi-Sensor Fusion Safety Monitoring - ML Prototype Platform

A comprehensive Streamlit-based machine learning development platform for multi-sensor fusion safety monitoring systems. This application provides an end-to-end pipeline for developing, training, and deploying AI models for caregiver alert systems.

## 🎯 Overview

This platform implements a complete ML development environment for safety monitoring systems that use multiple sensor modalities including PIR motion sensors, thermal cameras, radar, environmental sensors, audio sensors, and door sensors. The system is designed to detect trapped dependents and alert caregivers with high accuracy (≥99.8% as per TRS requirements).

## 🏗️ Architecture

### Core Components

- **Data Management**: TRS-compliant dataset generation with realistic sensor fusion scenarios
- **Model Development**: Multiple neural network architectures (LSTM, Transformer, Multi-Modal Fusion)
- **Training Pipeline**: Comprehensive training with early stopping, cross-validation, and hyperparameter optimization
- **Model Evaluation**: Detailed performance metrics and statistical analysis
- **ONNX Export**: Edge deployment optimization for Raspberry Pi 5 and AI HAT+
- **Model Interpretation**: SHAP analysis, attention visualization, and feature importance
- **Experiment Tracking**: MLflow integration for experiment management

### Sensor Array

| Sensor Type | Model | Purpose | Features |
|-------------|--------|---------|----------|
| PIR Motion | HC-SR501 | Motion detection | 8 statistical/time domain features |
| Thermal Camera | AMG8833 | Presence detection | 8x8 thermal grid analysis |
| 60GHz Radar | BGT60TR13C | Vital signs, motion | Range, velocity, signal strength |
| Environmental | DS18B20+, SHT40, SCD40 | Context monitoring | Temperature, humidity, CO2 |
| Audio Array | INMP441 | Sound analysis | 4-mic spatial audio processing |
| Door Sensors | MC-38, A3144 | Access monitoring | Magnetic/Hall effect sensing |

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sensor-fusion-ml-platform
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if using uv:
   ```bash
   uv sync
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the platform**
   - Open your browser to `http://localhost:8501`
   - The platform will be ready for ML development

### First Steps

1. **Generate Dataset**: Start with the "Data Management" tab to create TRS-compliant synthetic data
2. **Preprocess Data**: Configure normalization and feature engineering in "Preprocessing"
3. **Develop Models**: Choose architecture and fusion techniques in "Model Development"
4. **Train & Evaluate**: Monitor training progress and evaluate performance
5. **Export Models**: Generate ONNX models for edge deployment
6. **Interpret Results**: Analyze model behavior and feature importance

## 🧠 Model Architectures

### LSTM Networks
- **Bidirectional LSTM** with attention mechanism
- **Configurable layers** (1-6 layers)
- **Hidden dimensions** (32-512 units)
- **Dropout regularization** for overfitting prevention

### Transformer Models
- **Multi-head attention** for sequence modeling
- **Positional encoding** for temporal awareness
- **Layer normalization** and residual connections
- **Scalable architecture** (2-8 layers, 4-16 heads)

### Multi-Modal Fusion
- **Sensor-specific encoders** for each modality
- **Cross-modal attention** mechanisms
- **Dynamic fusion** with learnable weights
- **Hierarchical fusion** strategies

## 🔄 Fusion Techniques

### Early Fusion
- Concatenates raw sensor features
- Simple but effective baseline
- Normalizes across sensor modalities

### Feature-Level Fusion
- Extracts features per sensor type
- Applies sensor-specific processing
- Combines at feature level

### Late Fusion
- Independent sensor processing
- Combines predictions/decisions
- Weighted voting mechanisms

### Dynamic Gated Fusion
- Adaptive sensor weighting
- Input-dependent gating networks
- Quality-aware fusion decisions

## 📊 Performance Metrics

### Classification Metrics
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Per-class performance analysis
- **F1-Score**: Balanced performance measure
- **Confusion Matrix**: Detailed error analysis

### Risk Assessment
- **Risk Score Regression**: Continuous risk estimation
- **Confidence Analysis**: Prediction certainty measures
- **Scenario-Specific Performance**: Per-scenario accuracy

### Edge Deployment
- **Inference Time**: Real-time performance measurement
- **Model Size**: Memory footprint analysis
- **Power Consumption**: Energy efficiency metrics

## 🔧 Configuration

### Sensor Specifications
All sensor configurations follow TRS technical requirements:
- **PIR Sensors**: HC-SR501, 3-7m range, 120° detection angle
- **Thermal Camera**: AMG8833, 8×8 resolution, 60° FOV
- **Radar Module**: BGT60TR13C, 57-64 GHz, 0.15-10m range
- **Environmental**: Multi-sensor array for context
- **Audio Array**: 4× INMP441 microphones, 60Hz-15kHz
- **Door Sensors**: Magnetic + Hall effect sensing

### Model Configurations
Predefined configurations for different use cases:
- **Small Models**: Edge deployment, <10MB
- **Medium Models**: Balanced performance/size
- **Large Models**: Maximum accuracy, cloud deployment

## 🎛️ Hyperparameter Optimization

### Optuna Integration
- **Automated tuning** of learning rates, architectures
- **Multi-objective optimization** for accuracy vs efficiency
- **Pruning strategies** for faster convergence
- **Visualization** of optimization progress

### Grid Search Options
- **Learning Rate**: 1e-5 to 1e-2 (log scale)
- **Batch Size**: 16, 32, 64, 128
- **Architecture Depth**: 1-6 layers
- **Dropout Rates**: 0.0-0.5
- **Fusion Parameters**: Technique-specific tuning

## 📦 ONNX Export & Deployment

### Edge Optimization
- **Raspberry Pi 5**: INT8 quantization, <50ms inference
- **AI HAT+**: FP16 precision, <5ms inference with Hailo-8
- **Generic Edge**: Optimized for ARM processors

### Export Features
- **Dynamic batching** for variable input sizes
- **Model optimization** with constant folding
- **Quantization** for reduced memory usage
- **Validation** against PyTorch models

### Deployment Packages
- **Metadata files** with deployment instructions
- **Performance benchmarks** for target hardware
- **Installation scripts** for edge devices

## 🔍 Model Interpretability

### SHAP Analysis
- **Feature importance** across all sensors
- **Waterfall plots** for individual predictions
- **Summary visualizations** for model behavior

### Attention Visualization
- **Temporal attention** weights for sequence models
- **Cross-modal attention** for fusion analysis
- **Interactive heatmaps** for attention patterns

### Sensor Importance Analysis
- **Ablation studies** for sensor contribution
- **Reliability scoring** based on data quality
- **Fusion weight optimization** for best performance

## 🧪 Experiment Tracking

### MLflow Integration
- **Experiment logging** with automatic versioning
- **Parameter tracking** for reproducibility
- **Metric comparison** across model variants
- **Model registry** for production deployment

### Version Control
- **Model checkpoints** with performance metadata
- **Configuration snapshots** for experiment reproduction
- **Result archiving** for long-term analysis

## 🏥 Safety & Compliance

### TRS Compliance
- **99.8% minimum accuracy** requirement
- **Real-time processing** (<3s response time)
- **Fail-safe operation** during system failures
- **Power management** (24h battery backup)

### Data Privacy
- **Local processing** for sensitive data
- **Encrypted storage** for model parameters
- **Anonymized datasets** for training
- **GDPR compliance** for data handling

## 🚨 Alert System Integration

### Risk Classification
- **Low Risk**: Normal activity monitoring
- **Medium Risk**: Elevated attention required
- **High Risk**: Caregiver intervention needed
- **Critical Risk**: Emergency response required

### Communication Channels
- **SMS/Text**: Immediate caregiver alerts
- **Push Notifications**: Mobile app integration
- **Email**: Detailed incident reports
- **Emergency Services**: Critical situation escalation

## 🔬 Research & Development

### Ongoing Improvements
- **Transfer learning** from pre-trained models
- **Federated learning** for privacy-preserving training
- **Continual learning** for adaptation to new environments
- **Multi-task learning** for related safety applications

### Future Enhancements
- **Computer vision** integration for visual monitoring
- **NLP processing** for voice command analysis
- **IoT integration** with smart home systems
- **Cloud deployment** for scalable processing

## 📚 Documentation

### API Reference
- **Model Classes**: Detailed architecture documentation
- **Fusion Techniques**: Implementation guides
- **Evaluation Metrics**: Performance measurement details
- **Export Utilities**: ONNX conversion processes

### Examples & Tutorials
- **Getting Started**: Step-by-step platform tour
- **Model Development**: Architecture selection guide
- **Deployment**: Edge device setup instructions
- **Troubleshooting**: Common issues and solutions

### Dependencies & Requirements

#### System Requirements
- **Python**: 3.11 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for training)
- **Storage**: 2GB free space for models and datasets
- **OS**: Windows, macOS, or Linux

#### Core Dependencies
- **PyTorch**: Deep learning framework for model development
- **Streamlit**: Web application framework for the user interface
- **scikit-learn**: Machine learning utilities and evaluation metrics
- **NumPy/Pandas**: Data manipulation and numerical computing
- **Matplotlib/Seaborn/Plotly**: Visualization and plotting

#### Installation Tips
- **Virtual Environment**: Always recommended to avoid dependency conflicts
- **PyTorch**: CPU version sufficient for development; GPU version for faster training
- **uv**: Alternative package manager for faster dependency resolution
- **Development Tools**: Optional pytest, black, flake8 for code quality

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Run tests and validation
5. Submit pull request with documentation

### Code Standards
- **Type hints** for all function parameters
- **Docstrings** following Google style
- **Unit tests** for core functionality
- **Integration tests** for end-to-end workflows

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For technical support and questions:
- **Documentation**: Check inline help and tooltips
- **Issues**: Submit GitHub issues for bugs/features
- **Community**: Join discussions for best practices
- **Commercial**: Contact for enterprise deployment support

---

**Note**: This platform is designed for research and development purposes. For production deployment in safety-critical applications, ensure proper validation, testing, and regulatory compliance according to local requirements.
