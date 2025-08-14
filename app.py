import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# Core ML imports
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    st.error("PyTorch not available. Install torch for full functionality.")

# Visualization imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ML utilities
try:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional advanced libraries
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import onnx
    import onnxruntime
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# Import custom modules with error handling
try:
    from src.data_loader import TRSDataLoader
    from src.models import SensorFusionLSTM, SensorFusionTransformer, MultiModalFusionNet, get_model
    from src.fusion_techniques import get_fusion_technique, EarlyFusion, FeatureLevelFusion, LateFusion, DynamicGatedFusion
    from src.preprocessing import SensorDataPreprocessor
    from src.evaluation import ModelEvaluator
    from src.onnx_export import ONNXExporter
    from src.interpretability import ModelInterpreter
    from utils.constants import ScenarioType, RiskLevel, SENSOR_TYPES, MODEL_CONFIGS
    HAS_CUSTOM_MODULES = True
except ImportError as e:
    HAS_CUSTOM_MODULES = False
    st.error(f"Custom modules not fully available: {e}")
    # Define fallback classes and constants
    class MockDataLoader:
        def __init__(self): pass
        def generate_dataset(self, *args, **kwargs): return None, None, None, None
    
    TRSDataLoader = MockDataLoader
    # Create mock enums with proper attributes
    class ScenarioType:
        NORMAL_ACTIVITY = 0
        FALL_DETECTED = 1
        MEDICAL_EMERGENCY = 2
        NO_MOVEMENT = 3
        TRAPPED = 1
        EMERGENCY = 2
        MAINTENANCE = 3
        
        @classmethod
        def __iter__(cls):
            return iter([cls.NORMAL_ACTIVITY, cls.FALL_DETECTED, cls.MEDICAL_EMERGENCY, cls.NO_MOVEMENT])
    
    class RiskLevel:
        LOW = 0
        MEDIUM = 1
        HIGH = 2
        CRITICAL = 3
        
        @classmethod
        def __iter__(cls):
            return iter([cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL])
        
        @classmethod
        def __len__(cls):
            return 4
    SENSOR_TYPES = ['PIR', 'Thermal', 'Radar', 'Environmental', 'Audio', 'Door']
    MODEL_CONFIGS = {}

# Set page config
st.set_page_config(
    page_title="ML Prototype Development Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'experiment_history' not in st.session_state:
    st.session_state.experiment_history = []

def main():
    st.title("🔬 Multi-Sensor Fusion Safety Monitoring")
    st.markdown("### ML Prototype Development Platform")
    
    # Add stock photo gallery
    with st.container():
        st.markdown("#### System Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image("https://pixabay.com/get/g122395290cf51b326c6aed01fdedb0d8f5fa6c2da9d5ae9036a9834d81dc9fc29df8e6b472f5d4315a8f7d9ba3acf6568b93f1bfd4e24f416fd5337b3e1040d6_1280.jpg", 
                     caption="IoT Sensor Network", use_container_width=True)
        
        with col2:
            st.image("https://pixabay.com/get/g598205ab9e1fc7f69ac5bf75c13e8219b47570e1fc3bb4faea82ea5b8b94b367a2e499e85901e8c75dcba5dcaa32dab181392b97c2019194c66e759fe2fa91bc_1280.jpg", 
                     caption="Machine Learning Pipeline", use_container_width=True)
        
        with col3:
            st.image("https://pixabay.com/get/g7f7928dec789131e32fe33f04b556009b987dbd9c6418e06bbccdfbe3ff66b06478dca85ae1329fa3257c11d9173982bf8496fd9f5996418454885aec0eaebb5_1280.jpg", 
                     caption="Safety Monitoring Dashboard", use_container_width=True)
    
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        tab = st.radio(
            "Select Module",
            ["📊 Data Management", "🔧 Preprocessing", "🧠 Model Development", 
             "🏋️ Training & Evaluation", "📦 ONNX Export", "🔍 Model Interpretation", 
             "📈 Experiment Tracking"]
        )
        
        st.markdown("---")
        st.subheader("📈 System Status")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if st.session_state.data_loaded:
                st.success("✅ Data Ready")
            else:
                st.warning("⏳ No Data")
        
        with status_col2:
            if st.session_state.model_trained:
                st.success("✅ Model Ready")
            else:
                st.warning("⏳ No Model")
        
        if st.session_state.current_model:
            st.info(f"🤖 **Current Model:** {st.session_state.current_model}")
        
        # Additional sensor gallery
        st.markdown("---")
        st.subheader("🔬 Sensor Technologies")
        sensor_images = [
            "https://pixabay.com/get/gd10cf5ae29b16296969311b432fd29f8016bef8afca31ae642d3b06f7316292683eb5e00e5f8996ceaca09a51352e73139101cdff5b2da1065c06963d6ca3723_1280.jpg",
            "https://pixabay.com/get/g82c3e4968c811a84a122e9502256ed72e9e9431c77927cd5b4501a12098a6a5ce5529de91189fb7310d645f28ba5c980b4f8f06c23de3ca72dd6fda2fa44156e_1280.jpg",
            "https://pixabay.com/get/g2922a99b5d7ab7d29966c57fbd531c55359b896595146058df486dd5fcc8ff3e7f2743dcd1ab639554ff50faf9e7a2a796a45754b31584ad14d82a2eff0adf62_1280.jpg"
        ]
        
        for img_url in sensor_images:
            st.image(img_url, use_container_width=True)
    
    # Main content based on selected tab
    tab_key = tab.split(" ", 1)[1]  # Remove emoji prefix
    
    if tab_key == "Data Management":
        data_management_page()
    elif tab_key == "Preprocessing":
        preprocessing_page()
    elif tab_key == "Model Development":
        model_development_page()
    elif tab_key == "Training & Evaluation":
        training_evaluation_page()
    elif tab_key == "ONNX Export":
        onnx_export_page()
    elif tab_key == "Model Interpretation":
        interpretation_page()
    elif tab_key == "Experiment Tracking":
        experiment_tracking_page()

def data_management_page():
    st.header("📊 Data Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Dataset Generation")
        
        # Dataset parameters
        with st.expander("⚙️ Dataset Configuration", expanded=True):
            col1a, col1b = st.columns(2)
            with col1a:
                num_scenarios = st.number_input("Number of Scenarios", min_value=100, max_value=10000, value=1000)
                duration_hours = st.slider("Scenario Duration (hours)", 0.1, 24.0, (1.0, 8.0))
                
            with col1b:
                scenario_types = st.multiselect(
                    "Scenario Types",
                    options=[s.value for s in ScenarioType],
                    default=[ScenarioType.NORMAL_ACTIVITY.value, ScenarioType.FALL_DETECTED.value, 
                            ScenarioType.MEDICAL_EMERGENCY.value, ScenarioType.NO_MOVEMENT.value]
                )
                seed = st.number_input("Random Seed", value=42)
        
        if st.button("🚀 Generate Dataset", type="primary", use_container_width=True):
            with st.spinner("Generating TRS-compliant dataset..."):
                try:
                    data_loader = TRSDataLoader(seed=seed)
                    dataset = data_loader.generate_comprehensive_dataset(
                        num_scenarios=num_scenarios,
                        duration_range=duration_hours,
                        scenario_types=[ScenarioType(s) for s in scenario_types]
                    )
                    
                    st.session_state.dataset = dataset
                    st.session_state.data_loaded = True
                    st.success(f"✅ Dataset generated successfully! {len(dataset)} samples created.")
                    
                    # Display dataset statistics
                    st.subheader("📊 Dataset Statistics")
                    stats_df = data_loader.get_dataset_statistics(dataset)
                    st.dataframe(stats_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error generating dataset: {str(e)}")
    
    with col2:
        st.subheader("🔧 Sensor Specifications")
        
        sensor_type = st.selectbox("Select Sensor", SENSOR_TYPES)
        
        # Display more detailed sensor specs from utils/sensor_specs.py
        from utils.sensor_specs import get_sensor_specifications
        
        try:
            sensor_specs = get_sensor_specifications(sensor_type)
            st.json(sensor_specs)
        except Exception as e:
            # Fallback to basic specs
            if sensor_type == "PIR":
                st.json({
                    "model": "HC-SR501",
                    "detection_range_m": "3-7",
                    "detection_angle_deg": 120,
                    "operating_voltage": "4.5-20V",
                    "power_consumption_mw": 0.165
                })
            elif sensor_type == "Thermal":
                st.json({
                    "model": "AMG8833",
                    "resolution": "8x8",
                    "field_of_view_deg": "60x60",
                    "temp_range_c": "0-80",
                    "power_consumption_mw": 14.85
                })
            elif sensor_type == "Radar":
                st.json({
                    "model": "BGT60TR13C",
                    "frequency_range_ghz": "57-64",
                    "detection_range_m": "0.15-10",
                    "power_consumption_mw": 280.5
                })
    
    # Display dataset if loaded
    if st.session_state.data_loaded and 'dataset' in st.session_state:
        st.markdown("---")
        st.subheader("👀 Dataset Preview")
        
        # Convert dataset to DataFrame for display
        df_preview = pd.DataFrame([
            {
                'scenario': item['scenario'].value,
                'risk_level': item['risk_level'].value,
                'duration_h': item['duration_h'],
                'num_sensors': len(item['sensor_readings']),
                'timestamp': item['timestamp']
            }
            for item in st.session_state.dataset[:100]  # Show first 100 samples
        ])
        
        st.dataframe(df_preview, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Scenario distribution
            scenario_counts = df_preview['scenario'].value_counts()
            fig = px.pie(values=scenario_counts.values, names=scenario_counts.index, 
                        title="📊 Scenario Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Risk level distribution
            risk_counts = df_preview['risk_level'].value_counts()
            colors = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'}
            fig = px.bar(x=risk_counts.index, y=risk_counts.values, 
                        title="⚠️ Risk Level Distribution",
                        color=risk_counts.index,
                        color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)

def preprocessing_page():
    st.header("🔧 Data Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load dataset first from the Data Management page.")
        return
    
    st.subheader("⚙️ Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔄 Normalization Options**")
        normalization_method = st.selectbox(
            "Normalization Method",
            ["z-score", "min-max", "robust", "none"]
        )
        
        remove_outliers = st.checkbox("🚫 Remove Outliers", value=True)
        outlier_threshold = st.slider("Outlier Threshold (IQR multiplier)", 1.0, 3.0, 1.5)
        
    with col2:
        st.write("**🛠️ Feature Engineering**")
        window_size = st.number_input("Time Window Size", min_value=5, max_value=100, value=30)
        overlap = st.slider("Window Overlap (%)", 0, 90, 50)
        
        feature_types = st.multiselect(
            "Feature Types",
            ["statistical", "frequency", "time_domain", "wavelet"],
            default=["statistical", "time_domain"]
        )
    
    if st.button("🚀 Apply Preprocessing", type="primary", use_container_width=True):
        with st.spinner("Processing data..."):
            try:
                preprocessor = SensorDataPreprocessor(
                    normalization_method=normalization_method,
                    remove_outliers=remove_outliers,
                    outlier_threshold=outlier_threshold,
                    window_size=window_size,
                    overlap=overlap / 100,
                    feature_types=feature_types
                )
                
                processed_data = preprocessor.preprocess_dataset(st.session_state.dataset)
                st.session_state.processed_data = processed_data
                
                st.success("✅ Preprocessing completed successfully!")
                
                # Display preprocessing statistics
                st.subheader("📊 Preprocessing Statistics")
                stats = preprocessor.get_preprocessing_stats()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Samples", stats['original_samples'])
                with col2:
                    st.metric("Processed Samples", stats['processed_samples'])
                with col3:
                    st.metric("Feature Dimensions", stats['feature_dimensions'])
                
                # Visualization of processed data
                st.subheader("📈 Data Visualization")
                
                if 'processed_features' in processed_data:
                    # Feature correlation heatmap
                    features_sample = processed_data['processed_features'][:1000]  # Sample for visualization
                    if len(features_sample) > 0:
                        # Handle 3D data (samples, timesteps, features) -> 2D (samples, features)
                        if features_sample.ndim == 3:
                            features_sample = features_sample.reshape(-1, features_sample.shape[-1])
                        
                        # Calculate correlation for a subset of features
                        n_features = min(20, features_sample.shape[1])
                        correlation_matrix = np.corrcoef(features_sample[:, :n_features].T)
                        
                        fig = px.imshow(correlation_matrix, 
                                      title="🔗 Feature Correlation Matrix (Top 20 Features)",
                                      color_continuous_scale='RdBu',
                                      aspect='auto')
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Preprocessing error: {str(e)}")
                st.exception(e)

def model_development_page():
    st.header("🧠 Model Development")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load and preprocess dataset first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏗️ Model Architecture Selection")
        
        model_type = st.selectbox(
            "Model Type",
            ["LSTM", "Transformer", "Multi-Modal Fusion", "Deep Residual"]
        )
        
        fusion_technique = st.selectbox(
            "Fusion Technique",
            ["Early Fusion", "Feature-Level Fusion", "Late Fusion", "Dynamic Gated Fusion"]
        )
        
        # Model-specific parameters
        with st.expander("⚙️ Model Configuration", expanded=True):
            if model_type == "LSTM":
                hidden_size = st.number_input("Hidden Size", min_value=32, max_value=512, value=128)
                num_layers = st.number_input("Number of Layers", min_value=1, max_value=6, value=2)
                dropout = st.slider("Dropout Rate", 0.0, 0.7, 0.2)
                bidirectional = st.checkbox("Bidirectional", value=True)
                
            elif model_type == "Transformer":
                d_model = st.number_input("Model Dimension", min_value=64, max_value=512, value=256)
                nhead = st.selectbox("Number of Heads", [4, 8, 12, 16], index=1)
                num_layers = st.number_input("Number of Layers", min_value=2, max_value=8, value=4)
                dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.1)
                
            elif model_type == "Multi-Modal Fusion":
                sensor_encoders = st.multiselect(
                    "Sensor Encoders",
                    ["PIR", "Thermal", "Radar", "Environmental", "Audio", "Door"],
                    default=["PIR", "Thermal", "Radar"]
                )
                encoder_dim = st.number_input("Encoder Dimension", min_value=64, max_value=256, value=128)
                fusion_dim = st.number_input("Fusion Dimension", min_value=128, max_value=512, value=256)
                
            else:  # Deep Residual
                hidden_dim = st.number_input("Hidden Dimension", min_value=64, max_value=512, value=256)
                num_blocks = st.number_input("Number of Blocks", min_value=2, max_value=8, value=4)
                dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        
        # Training parameters
        with st.expander("🏋️ Training Configuration"):
            learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)
            epochs = st.number_input("Epochs", min_value=10, max_value=200, value=50)
            optimizer = st.selectbox("Optimizer", ["Adam", "AdamW", "SGD"], index=1)
            scheduler = st.selectbox("Scheduler", ["None", "StepLR", "CosineAnnealingLR"], index=1)
    
    with col2:
        st.subheader("📋 Model Summary")
        
        # Display current model configuration
        config = {
            "Model Type": model_type,
            "Fusion Technique": fusion_technique,
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
            "Epochs": epochs
        }
        
        for key, value in config.items():
            st.write(f"**{key}:** {value}")
        
        st.markdown("---")
        
        # Model architecture visualization placeholder
        st.image("https://pixabay.com/get/gdff0736014871f6d004835ed22bc534df680013cff8896c2b033b7d2ebd23ad4d1e3b3356e20e1e84fd3390013cd0390fb79b096430e69948e7fa5138bebba57_1280.jpg", 
                 caption="Neural Network Architecture", use_container_width=True)
    
    if st.button("🚀 Initialize Model", type="primary", use_container_width=True):
        with st.spinner("Initializing model..."):
            try:
                # Initialize fusion technique
                fusion_mapping = {
                    "Early Fusion": "early",
                    "Feature-Level Fusion": "feature",
                    "Late Fusion": "late",
                    "Dynamic Gated Fusion": "dynamic"
                }
                
                fusion = get_fusion_technique(
                    fusion_mapping[fusion_technique],
                    input_dim=39  # Total feature dimensions
                )
                
                # Initialize model based on type
                if model_type == "LSTM":
                    model = SensorFusionLSTM(
                        input_size=39,  # Total features from all sensors
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        num_classes=len(RiskLevel),
                        dropout=dropout,
                        bidirectional=bidirectional,
                        fusion_technique=fusion
                    )
                elif model_type == "Transformer":
                    model = SensorFusionTransformer(
                        input_size=39,
                        d_model=d_model,
                        nhead=nhead,
                        num_layers=num_layers,
                        num_classes=len(RiskLevel),
                        dropout=dropout,
                        fusion_technique=fusion
                    )
                elif model_type == "Multi-Modal Fusion":
                    model = MultiModalFusionNet(
                        sensor_types=sensor_encoders,
                        encoder_dim=encoder_dim,
                        fusion_dim=fusion_dim,
                        num_classes=len(RiskLevel),
                        fusion_technique=fusion
                    )
                else:  # Deep Residual
                    from src.models import DeepSensorFusionNet
                    model = DeepSensorFusionNet(
                        input_size=39,
                        hidden_dim=hidden_dim,
                        num_blocks=num_blocks,
                        num_classes=len(RiskLevel),
                        dropout=dropout,
                        fusion_technique=fusion
                    )
                
                st.session_state.model = model
                st.session_state.current_model = f"{model_type}_{fusion_technique}"
                st.session_state.model_config = config
                
                st.success("✅ Model initialized successfully!")
                
                # Display model information
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Parameters", f"{total_params:,}")
                with col2:
                    st.metric("Trainable Parameters", f"{trainable_params:,}")
                with col3:
                    st.metric("Model Size (MB)", f"{total_params * 4 / (1024**2):.2f}")
                
            except Exception as e:
                st.error(f"❌ Model initialization error: {str(e)}")
                st.exception(e)

def training_evaluation_page():
    st.header("🏋️ Training & Evaluation")
    
    if not hasattr(st.session_state, 'model'):
        st.warning("⚠️ Please initialize a model first from the Model Development page.")
        return
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📈 Training Progress")
        
        # Training controls
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                train_model()
        with col1b:
            if st.button("⏸️ Pause Training", use_container_width=True):
                st.info("Training paused")
        with col1c:
            if st.button("⏹️ Stop Training", use_container_width=True):
                st.info("Training stopped")
        
        # Training metrics display
        if st.session_state.model_trained and 'training_history' in st.session_state:
            history = st.session_state.training_history
            
            # Loss plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('📉 Training Loss', '📉 Validation Loss', 
                              '📊 Training Accuracy', '📊 Validation Accuracy')
            )
            
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], 
                                   name='Train Loss', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], 
                                   name='Val Loss', line=dict(color='red')), row=1, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=history['train_acc'], 
                                   name='Train Acc', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history['val_acc'], 
                                   name='Val Acc', line=dict(color='orange')), row=2, col=2)
            
            fig.update_layout(height=600, title_text="📊 Training Metrics", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Model Performance")
        
        if st.session_state.model_trained and 'evaluation_results' in st.session_state:
            results = st.session_state.evaluation_results
            
            # Key metrics
            st.metric("🎯 Overall Accuracy", f"{results['accuracy']:.3f}")
            st.metric("📊 F1 Score (Macro)", f"{results['f1_macro']:.3f}")
            st.metric("🔍 Precision (Macro)", f"{results['precision_macro']:.3f}")
            st.metric("🎪 Recall (Macro)", f"{results['recall_macro']:.3f}")
            
            st.markdown("---")
            
            # Per-class performance
            st.subheader("📋 Per-Class Performance")
            st.text(results['classification_report'])
            
        else:
            st.info("🏋️ Train model to see performance metrics")
            st.image("https://pixabay.com/get/g88bf8db9e3270f14688016e6973c44b9edf83073d1008738d944664f79cbc5aef47eeca9cf8ca8647c9f73b233ad8e2a287abc4bc26421bea5b69ff9e190c9f9_1280.jpg", 
                     caption="Training Pipeline", use_container_width=True)
    
    # Hyperparameter optimization
    st.markdown("---")
    st.subheader("🔧 Hyperparameter Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_trials = st.number_input("Number of Trials", min_value=10, max_value=100, value=20)
        optimization_metric = st.selectbox("Optimization Metric", ["accuracy", "f1_macro", "val_loss"])
        
    with col2:
        study_name = st.text_input("Study Name", value=f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    if st.button("🚀 Start Hyperparameter Optimization", use_container_width=True):
        with st.spinner("Running hyperparameter optimization..."):
            run_hyperparameter_optimization(n_trials, optimization_metric, study_name)

def train_model():
    """Train the model with current configuration"""
    with st.spinner("Training model..."):
        try:
            # Prepare data
            if 'processed_data' in st.session_state:
                data = st.session_state.dataset  # Use original data, evaluator will process it
            else:
                # Use raw data if preprocessing not done
                data = st.session_state.dataset
            
            # Split data with stratification to ensure all classes are represented
            try:
                # Create stratification labels
                stratify_labels = [sample['risk_level'].value for sample in data]
                train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=stratify_labels)
                
                # Stratify train/val split as well
                train_stratify_labels = [sample['risk_level'].value for sample in train_data]
                train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_stratify_labels)
            except ValueError as e:
                # Fallback to non-stratified split if stratification fails
                st.warning(f"⚠️ Could not use stratified split: {e}. Using random split instead.")
                train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
                train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
            
            # Initialize trainer
            evaluator = ModelEvaluator(st.session_state.model)
            
            # Train model
            history = evaluator.train(
                train_data=train_data,
                val_data=val_data,
                epochs=st.session_state.model_config.get('Epochs', 50),
                batch_size=st.session_state.model_config.get('Batch Size', 64),
                learning_rate=st.session_state.model_config.get('Learning Rate', 1e-3)
            )
            
            # Evaluate model
            results = evaluator.evaluate(test_data)
            
            # Store results
            st.session_state.training_history = history
            st.session_state.evaluation_results = results
            st.session_state.model_trained = True
            
            # Add to experiment history
            experiment = {
                'timestamp': datetime.now(),
                'model_type': st.session_state.current_model,
                'config': st.session_state.model_config,
                'accuracy': results['accuracy'],
                'f1_score': results['f1_macro']
            }
            st.session_state.experiment_history.append(experiment)
            
            st.success("✅ Model training completed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            st.exception(e)

def run_hyperparameter_optimization(n_trials, metric, study_name):
    """Run hyperparameter optimization using Optuna"""
    try:
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            
            # Create and train model with suggested parameters
            fusion = get_fusion_technique("early", input_dim=39)
            model = SensorFusionLSTM(
                input_size=39,
                hidden_size=hidden_size,
                num_layers=2,
                num_classes=len(RiskLevel),
                dropout=dropout,
                fusion_technique=fusion
            )
            
            evaluator = ModelEvaluator(model)
            
            # Quick training for optimization
            data = st.session_state.dataset
            train_size = int(len(data) * 0.6)
            val_start = int(len(data) * 0.6)
            val_end = int(len(data) * 0.8)
            
            history = evaluator.train(
                train_data=data[:train_size],
                val_data=data[val_start:val_end],
                epochs=20,  # Reduced for optimization
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Return metric to optimize
            if metric == 'val_loss':
                return min(history['val_loss'])
            else:
                return max(history.get('val_acc', [0]))
        
        # Run optimization
        study = optuna.create_study(direction='minimize' if metric == 'val_loss' else 'maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Display results
        st.success("✅ Hyperparameter optimization completed!")
        
        best_params = study.best_params
        st.subheader("🏆 Best Parameters")
        st.json(best_params)
        
        st.subheader("📈 Optimization History")
        fig = px.line(x=range(len(study.trials)), 
                     y=[t.value for t in study.trials],
                     title="🔍 Optimization Progress",
                     labels={'x': 'Trial', 'y': 'Objective Value'})
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Optimization error: {str(e)}")
        st.exception(e)

def onnx_export_page():
    st.header("📦 ONNX Model Export")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first before exporting to ONNX.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Export Configuration")
        
        export_name = st.text_input("Model Name", value="sensor_fusion_model")
        opset_version = st.selectbox("ONNX Opset Version", [11, 12, 13, 14], index=0)
        optimize_model = st.checkbox("🚀 Optimize for Edge Deployment", value=True)
        quantize_model = st.checkbox("📏 Apply INT8 Quantization", value=False)
        
        # Target platform
        target_platform = st.selectbox(
            "🎯 Target Platform",
            ["Raspberry Pi 5", "Edge Device", "Generic", "AI HAT+"]
        )
        
        batch_size = st.number_input("Batch Size for Export", min_value=1, max_value=32, value=1)
        
    with col2:
        st.subheader("📊 Model Information")
        
        if hasattr(st.session_state, 'model'):
            model = st.session_state.model
            total_params = sum(p.numel() for p in model.parameters())
            
            st.metric("🔢 Total Parameters", f"{total_params:,}")
            st.metric("📏 Model Size (PyTorch)", f"{total_params * 4 / (1024**2):.2f} MB")
            
            # Estimated ONNX model size
            estimated_onnx_size = total_params * 4 / (1024**2)
            if quantize_model:
                estimated_onnx_size /= 4  # INT8 quantization
            
            st.metric("📦 Estimated ONNX Size", f"{estimated_onnx_size:.2f} MB")
        
        # Edge device image
        st.image("https://pixabay.com/get/gc918f444e91aed3a8e99eb091f4e386683ea1a288ac6ef5c1ca9a3bf6ee32e6187e372e63b1b796cb8716d5b3a02f9ec5c202429300b24b89b897d2264f8c424_1280.jpg", 
                 caption="Edge Deployment Target", use_container_width=True)
    
    if st.button("🚀 Export to ONNX", type="primary", use_container_width=True):
        with st.spinner("Exporting model to ONNX..."):
            try:
                exporter = ONNXExporter(
                    model=st.session_state.model,
                    opset_version=opset_version,
                    optimize=optimize_model,
                    quantize=quantize_model
                )
                
                # Export model
                onnx_path = exporter.export(
                    model_name=export_name,
                    batch_size=batch_size,
                    input_shape=(batch_size, 30, 39)  # Default shape
                )
                
                st.success(f"✅ Model exported successfully to {onnx_path}")
                
                # Model validation
                validation_results = exporter.validate_onnx_model(onnx_path)
                
                st.subheader("🔍 Export Validation")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("✅ Model Valid", "✅" if validation_results['valid'] else "❌")
                with col2:
                    st.metric("📐 Input Shape", str(validation_results.get('input_shape', 'N/A')))
                with col3:
                    st.metric("📐 Output Shape", str(validation_results.get('output_shape', 'N/A')))
                
                # Performance comparison
                if validation_results['valid']:
                    st.subheader("⚡ Performance Comparison")
                    
                    perf_results = exporter.benchmark_models(
                        pytorch_model=st.session_state.model,
                        onnx_path=onnx_path,
                        num_runs=100
                    )
                    
                    comparison_df = pd.DataFrame([
                        {"Model": "PyTorch", 
                         "Inference Time (ms)": f"{perf_results['pytorch_time']:.2f}", 
                         "Memory (MB)": f"{perf_results['pytorch_memory']:.2f}"},
                        {"Model": "ONNX", 
                         "Inference Time (ms)": f"{perf_results['onnx_time']:.2f}", 
                         "Memory (MB)": f"{perf_results['onnx_memory']:.2f}"}
                    ])
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Speedup calculation
                    speedup = perf_results['pytorch_time'] / perf_results['onnx_time']
                    st.metric("🚀 Speedup Factor", f"{speedup:.2f}x")
                
            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")
                st.exception(e)

def interpretation_page():
    st.header("🔍 Model Interpretation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first before running interpretation analysis.")
        return
    
    st.subheader("🧠 Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        interpretation_method = st.selectbox(
            "🔬 Interpretation Method",
            ["SHAP", "Integrated Gradients", "Layer-wise Relevance Propagation", "Attention Weights"]
        )
        
        num_samples = st.number_input("Number of Samples for Analysis", min_value=10, max_value=1000, value=100)
        
    with col2:
        visualization_type = st.selectbox(
            "📊 Visualization Type",
            ["Feature Importance", "Sample Explanation", "Sensor Contribution", "Time Series Importance"]
        )
    
    if st.button("🚀 Run Interpretation Analysis", type="primary", use_container_width=True):
        with st.spinner("Running model interpretation..."):
            try:
                interpreter = ModelInterpreter(st.session_state.model)
                
                # Sample data for interpretation
                sample_data = st.session_state.dataset[:num_samples]
                
                if interpretation_method == "SHAP":
                    try:
                        results = interpreter.shap_analysis(sample_data)
                        
                        # SHAP summary plot
                        st.subheader("📊 SHAP Feature Importance")
                        fig = interpreter.plot_shap_summary(results)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # SHAP waterfall plot for single prediction
                        st.subheader("💧 Single Prediction Explanation")
                        waterfall_fig = interpreter.plot_shap_waterfall(results, sample_idx=0)
                        st.plotly_chart(waterfall_fig, use_container_width=True)
                    except ImportError:
                        st.error("❌ SHAP library not available. Please install with: pip install shap")
                
                elif interpretation_method == "Attention Weights":
                    if hasattr(st.session_state.model, 'get_attention_weights') or 'Transformer' in st.session_state.model.__class__.__name__:
                        attention_weights = interpreter.get_attention_analysis(sample_data)
                        
                        # Attention heatmap
                        st.subheader("🎯 Attention Weights Visualization")
                        fig = px.imshow(attention_weights[:50], 
                                      title="🔥 Attention Weights Across Time Steps",
                                      labels=dict(x="Time Step", y="Sample", color="Attention"),
                                      aspect='auto')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Current model doesn't support attention weight visualization.")
                
                # Sensor-specific analysis
                st.subheader("🔧 Sensor-Specific Importance")
                sensor_importance = interpreter.analyze_sensor_importance(sample_data)
                
                sensor_df = pd.DataFrame([
                    {"Sensor": sensor, "Importance": importance} 
                    for sensor, importance in sensor_importance.items()
                ])
                
                fig = px.bar(sensor_df, x="Sensor", y="Importance", 
                           title="🏆 Sensor Importance for Predictions",
                           color="Importance",
                           color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk scenario analysis
                st.subheader("⚠️ Risk Scenario Analysis")
                scenario_analysis = interpreter.analyze_risk_scenarios(sample_data)
                
                scenario_df = pd.DataFrame(scenario_analysis)
                st.dataframe(scenario_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Interpretation error: {str(e)}")
                st.exception(e)

def experiment_tracking_page():
    st.header("📈 Experiment Tracking")
    
    # MLflow integration
    st.subheader("🧪 Experiment History")
    
    if st.session_state.experiment_history:
        experiments_df = pd.DataFrame(st.session_state.experiment_history)
        
        # Experiment comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy over time
            fig = px.line(experiments_df, x='timestamp', y='accuracy', 
                         title="📊 Model Accuracy Over Time",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # F1 score comparison
            fig = px.bar(experiments_df, x='model_type', y='f1_score',
                        title="🎯 F1 Score by Model Type",
                        color='f1_score',
                        color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed experiment table
        st.subheader("📋 Experiment Details")
        st.dataframe(experiments_df, use_container_width=True)
        
        # Best model selection
        st.subheader("🏆 Best Performing Models")
        best_accuracy = experiments_df.loc[experiments_df['accuracy'].idxmax()]
        best_f1 = experiments_df.loc[experiments_df['f1_score'].idxmax()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🎯 Best Accuracy:**")
            st.json({k: str(v) for k, v in best_accuracy.to_dict().items()})
            
        with col2:
            st.write("**📊 Best F1 Score:**")
            st.json({k: str(v) for k, v in best_f1.to_dict().items()})
    
    else:
        st.info("📝 No experiments recorded yet. Train some models to see experiment tracking.")
        st.image("https://pixabay.com/get/g93551ba7ec1552a7bb120bd7810a11342917db258d2cff3a0783af198af768474b64ec9ddca72f4319e67d225441dc261e944ecf8e8138fea037918b75a7996c_1280.jpg", 
                 caption="Experiment Tracking System", use_container_width=True)
    
    # Model versioning
    st.markdown("---")
    st.subheader("📦 Model Versioning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        version_name = st.text_input("Version Name", value=f"v{len(st.session_state.experiment_history) + 1}")
        version_notes = st.text_area("Version Notes", placeholder="Describe changes or improvements...")
        
    with col2:
        if st.button("💾 Save Model Version", use_container_width=True):
            if st.session_state.model_trained:
                # Save model checkpoint
                checkpoint = {
                    'model_state_dict': st.session_state.model.state_dict(),
                    'model_config': st.session_state.model_config,
                    'version': version_name,
                    'notes': version_notes,
                    'timestamp': datetime.now(),
                    'performance': st.session_state.evaluation_results if hasattr(st.session_state, 'evaluation_results') else None
                }
                
                # Save to file
                os.makedirs('model_versions', exist_ok=True)
                torch.save(checkpoint, f'model_versions/{version_name}.pth')
                
                st.success(f"✅ Model version {version_name} saved successfully!")
            else:
                st.warning("⚠️ No trained model to save.")
    
    # Load saved models
    st.subheader("📁 Load Saved Models")
    
    if os.path.exists('model_versions'):
        saved_models = [f for f in os.listdir('model_versions') if f.endswith('.pth')]
        
        if saved_models:
            selected_model = st.selectbox("Select Model to Load", saved_models)
            
            if st.button("📂 Load Model", use_container_width=True):
                try:
                    checkpoint = torch.load(f'model_versions/{selected_model}', map_location='cpu')
                    
                    st.write("**📋 Model Information:**")
                    st.json({
                        'version': checkpoint['version'],
                        'timestamp': str(checkpoint['timestamp']),
                        'notes': checkpoint['notes']
                    })
                    
                    if checkpoint.get('performance'):
                        st.write("**🎯 Performance Metrics:**")
                        perf_metrics = {k: f"{v:.4f}" if isinstance(v, float) else str(v) 
                                      for k, v in checkpoint['performance'].items() 
                                      if k in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']}
                        st.json(perf_metrics)
                    
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
        else:
            st.info("📂 No saved models found.")

if __name__ == "__main__":
    main()
