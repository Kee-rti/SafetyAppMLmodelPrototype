import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import time
import os
import logging
from pathlib import Path
import tempfile
import warnings
warnings.filterwarnings('ignore')

class ONNXExporter:
    """
    ONNX model export pipeline for edge deployment optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        opset_version: int = 11,
        optimize: bool = True,
        quantize: bool = False,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ):
        """
        Initialize ONNX exporter
        
        Args:
            model: PyTorch model to export
            opset_version: ONNX opset version
            optimize: Whether to optimize the ONNX model
            quantize: Whether to apply INT8 quantization
            dynamic_axes: Dynamic axes for variable input sizes
        """
        self.model = model
        self.opset_version = opset_version
        self.optimize = optimize
        self.quantize = quantize
        self.dynamic_axes = dynamic_axes
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Export statistics
        self.export_stats = {}
    
    def export(
        self,
        model_name: str = "sensor_fusion_model",
        batch_size: int = 1,
        input_shape: Tuple[int, ...] = None,
        output_dir: str = "models",
        verbose: bool = False
    ) -> str:
        """
        Export PyTorch model to ONNX format
        
        Args:
            model_name: Name for the exported model
            batch_size: Batch size for export
            input_shape: Input shape (batch_size, seq_len, features)
            output_dir: Output directory for saved model
            verbose: Enable verbose logging
            
        Returns:
            Path to exported ONNX model
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine input shape
        if input_shape is None:
            # Default shape based on typical sensor fusion input
            input_shape = (batch_size, 30, 21)  # 30 timesteps, 21 features
        else:
            input_shape = (batch_size,) + input_shape[1:]
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Output path
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        
        self.logger.info(f"Exporting model to ONNX format...")
        self.logger.info(f"Input shape: {input_shape}")
        self.logger.info(f"Output path: {onnx_path}")
        
        try:
            # Determine input and output names
            input_names = ['sensor_data']
            
            # Check model output structure
            with torch.no_grad():
                sample_output = self.model(dummy_input)
                if isinstance(sample_output, tuple):
                    output_names = ['class_predictions', 'risk_scores']
                else:
                    output_names = ['predictions']
            
            # Setup dynamic axes if not provided
            if self.dynamic_axes is None:
                self.dynamic_axes = {
                    'sensor_data': {0: 'batch_size', 1: 'sequence_length'},
                    'class_predictions': {0: 'batch_size'},
                    'risk_scores': {0: 'batch_size'}
                }
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=self.dynamic_axes,
                verbose=verbose
            )
            
            self.logger.info("ONNX export completed successfully")
            
            # Verify exported model
            self._verify_onnx_model(onnx_path)
            
            # Optimize model if requested
            if self.optimize:
                optimized_path = self._optimize_onnx_model(onnx_path, model_name, output_dir)
                onnx_path = optimized_path
            
            # Quantize model if requested
            if self.quantize:
                quantized_path = self._quantize_onnx_model(onnx_path, model_name, output_dir)
                onnx_path = quantized_path
            
            # Calculate export statistics
            self._calculate_export_stats(onnx_path, input_shape)
            
            return onnx_path
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {str(e)}")
            raise
    
    def _verify_onnx_model(self, onnx_path: str) -> bool:
        """Verify the exported ONNX model"""
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            self.logger.info("ONNX model verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX model verification failed: {str(e)}")
            return False
    
    def _optimize_onnx_model(self, onnx_path: str, model_name: str, output_dir: str) -> str:
        """Optimize ONNX model for edge deployment"""
        try:
            import onnxoptimizer
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(
                onnx_model,
                passes=[
                    'eliminate_deadend',
                    'eliminate_identity',
                    'eliminate_nop_dropout',
                    'eliminate_nop_monotone_argmax',
                    'eliminate_nop_pad',
                    'eliminate_nop_transpose',
                    'eliminate_unused_initializer',
                    'extract_constant_to_initializer',
                    'fuse_add_bias_into_conv',
                    'fuse_bn_into_conv',
                    'fuse_consecutive_concats',
                    'fuse_consecutive_reduce_unsqueeze',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_transposes',
                    'fuse_matmul_add_bias_into_gemm',
                    'fuse_pad_into_conv',
                    'fuse_transpose_into_gemm'
                ]
            )
            
            # Save optimized model
            optimized_path = os.path.join(output_dir, f"{model_name}_optimized.onnx")
            onnx.save(optimized_model, optimized_path)
            
            self.logger.info(f"ONNX model optimized and saved to {optimized_path}")
            return optimized_path
            
        except ImportError:
            self.logger.warning("onnxoptimizer not available, skipping optimization")
            return onnx_path
        except Exception as e:
            self.logger.error(f"ONNX optimization failed: {str(e)}")
            return onnx_path
    
    def _quantize_onnx_model(self, onnx_path: str, model_name: str, output_dir: str) -> str:
        """Apply INT8 quantization to ONNX model"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Output path for quantized model
            quantized_path = os.path.join(output_dir, f"{model_name}_quantized.onnx")
            
            # Apply dynamic quantization
            quantize_dynamic(
                onnx_path,
                quantized_path,
                weight_type=QuantType.QInt8
            )
            
            self.logger.info(f"ONNX model quantized and saved to {quantized_path}")
            return quantized_path
            
        except ImportError:
            self.logger.warning("onnxruntime quantization not available, skipping quantization")
            return onnx_path
        except Exception as e:
            self.logger.error(f"ONNX quantization failed: {str(e)}")
            return onnx_path
    
    def _calculate_export_stats(self, onnx_path: str, input_shape: Tuple[int, ...]):
        """Calculate export statistics"""
        try:
            # Get file size
            file_size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
            
            # Load ONNX model for analysis
            onnx_model = onnx.load(onnx_path)
            
            # Count parameters
            total_params = 0
            for initializer in onnx_model.graph.initializer:
                param_size = 1
                for dim in initializer.dims:
                    param_size *= dim
                total_params += param_size
            
            self.export_stats = {
                'file_size_mb': file_size_mb,
                'total_parameters': total_params,
                'input_shape': input_shape,
                'opset_version': self.opset_version,
                'optimized': self.optimize,
                'quantized': self.quantize
            }
            
            self.logger.info(f"Export stats: {self.export_stats}")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate export stats: {str(e)}")
    
    def validate_onnx_model(self, onnx_path: str, num_test_samples: int = 10) -> Dict[str, Any]:
        """
        Validate ONNX model against PyTorch model
        
        Args:
            onnx_path: Path to ONNX model
            num_test_samples: Number of test samples for validation
            
        Returns:
            Validation results
        """
        self.logger.info("Validating ONNX model against PyTorch model...")
        
        try:
            # Load ONNX runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get input/output info
            input_info = ort_session.get_inputs()[0]
            input_shape = input_info.shape
            input_name = input_info.name
            
            # Replace dynamic dimensions with concrete values
            concrete_shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim is None:
                    if 'batch' in str(dim).lower():
                        concrete_shape.append(1)
                    elif 'sequence' in str(dim).lower():
                        concrete_shape.append(30)
                    else:
                        concrete_shape.append(10)
                else:
                    concrete_shape.append(dim)
            
            validation_results = {
                'valid': True,
                'input_shape': concrete_shape,
                'output_shape': None,
                'max_diff': 0.0,
                'mean_diff': 0.0,
                'inference_time_pytorch': 0.0,
                'inference_time_onnx': 0.0
            }
            
            # Test with random inputs
            total_pytorch_time = 0.0
            total_onnx_time = 0.0
            max_diff = 0.0
            mean_diffs = []
            
            for i in range(num_test_samples):
                # Generate random input
                test_input = np.random.randn(*concrete_shape).astype(np.float32)
                test_input_torch = torch.from_numpy(test_input)
                
                # PyTorch inference
                start_time = time.time()
                with torch.no_grad():
                    pytorch_output = self.model(test_input_torch)
                pytorch_time = time.time() - start_time
                total_pytorch_time += pytorch_time
                
                # Handle tuple outputs
                if isinstance(pytorch_output, tuple):
                    pytorch_output = pytorch_output[0]  # Use first output for comparison
                
                pytorch_result = pytorch_output.numpy()
                
                # ONNX inference
                start_time = time.time()
                onnx_output = ort_session.run(None, {input_name: test_input})
                onnx_time = time.time() - start_time
                total_onnx_time += onnx_time
                
                onnx_result = onnx_output[0]  # First output
                
                # Store output shape
                if validation_results['output_shape'] is None:
                    validation_results['output_shape'] = list(onnx_result.shape)
                
                # Compare outputs
                diff = np.abs(pytorch_result - onnx_result)
                max_diff = max(max_diff, np.max(diff))
                mean_diffs.append(np.mean(diff))
            
            validation_results['max_diff'] = float(max_diff)
            validation_results['mean_diff'] = float(np.mean(mean_diffs))
            validation_results['inference_time_pytorch'] = total_pytorch_time / num_test_samples * 1000  # ms
            validation_results['inference_time_onnx'] = total_onnx_time / num_test_samples * 1000  # ms
            
            # Check if differences are acceptable
            if max_diff > 1e-3:
                validation_results['valid'] = False
                self.logger.warning(f"Large difference detected: {max_diff}")
            else:
                self.logger.info("ONNX validation passed")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"ONNX validation failed: {str(e)}")
            return {
                'valid': False,
                'error': str(e),
                'input_shape': None,
                'output_shape': None
            }
    
    def benchmark_models(
        self,
        pytorch_model: nn.Module,
        onnx_path: str,
        num_runs: int = 100,
        input_shape: Tuple[int, ...] = None
    ) -> Dict[str, float]:
        """
        Benchmark PyTorch vs ONNX model performance
        
        Args:
            pytorch_model: Original PyTorch model
            onnx_path: Path to ONNX model
            num_runs: Number of benchmark runs
            input_shape: Input shape for benchmarking
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Benchmarking models with {num_runs} runs...")
        
        if input_shape is None:
            input_shape = (1, 30, 21)  # Default shape
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        
        # Warm up
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        dummy_input_torch = torch.from_numpy(dummy_input)
        
        # Warm up runs
        for _ in range(10):
            with torch.no_grad():
                _ = pytorch_model(dummy_input_torch)
            _ = ort_session.run(None, {input_name: dummy_input})
        
        # Benchmark PyTorch
        pytorch_times = []
        for _ in range(num_runs):
            test_input = np.random.randn(*input_shape).astype(np.float32)
            test_input_torch = torch.from_numpy(test_input)
            
            start_time = time.time()
            with torch.no_grad():
                _ = pytorch_model(test_input_torch)
            pytorch_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Benchmark ONNX
        onnx_times = []
        for _ in range(num_runs):
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            start_time = time.time()
            _ = ort_session.run(None, {input_name: test_input})
            onnx_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        results = {
            'pytorch_time': np.mean(pytorch_times),
            'pytorch_std': np.std(pytorch_times),
            'pytorch_memory': self._estimate_memory_usage(pytorch_model),
            'onnx_time': np.mean(onnx_times),
            'onnx_std': np.std(onnx_times),
            'onnx_memory': os.path.getsize(onnx_path) / (1024 ** 2),  # MB
            'speedup': np.mean(pytorch_times) / np.mean(onnx_times),
            'memory_reduction': (
                self._estimate_memory_usage(pytorch_model) - 
                os.path.getsize(onnx_path) / (1024 ** 2)
            )
        }
        
        self.logger.info(f"Benchmark results: {results}")
        return results
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate PyTorch model memory usage in MB"""
        total_params = sum(p.numel() for p in model.parameters())
        # Assume float32 (4 bytes per parameter)
        memory_mb = total_params * 4 / (1024 ** 2)
        return memory_mb
    
    def export_for_edge_device(
        self,
        target_platform: str = "raspberry_pi",
        model_name: str = "sensor_fusion_edge",
        output_dir: str = "edge_models"
    ) -> str:
        """
        Export model optimized for specific edge devices
        
        Args:
            target_platform: Target edge platform
            model_name: Model name
            output_dir: Output directory
            
        Returns:
            Path to optimized model
        """
        self.logger.info(f"Exporting model for {target_platform}")
        
        # Platform-specific optimizations
        if target_platform.lower() == "raspberry_pi":
            # Raspberry Pi optimizations
            self.opset_version = 11  # Better ARM support
            self.optimize = True
            self.quantize = True
            input_shape = (1, 30, 21)  # Single batch for edge
            
        elif target_platform.lower() == "ai_hat":
            # AI HAT+ optimizations
            self.opset_version = 13
            self.optimize = True
            self.quantize = False  # Hailo-8 handles quantization
            input_shape = (1, 30, 21)
            
        else:
            # Generic edge device
            self.optimize = True
            self.quantize = True
            input_shape = (1, 30, 21)
        
        # Export with platform-specific settings
        exported_path = self.export(
            model_name=f"{model_name}_{target_platform}",
            batch_size=1,
            input_shape=input_shape,
            output_dir=output_dir
        )
        
        # Create deployment package
        self._create_deployment_package(exported_path, target_platform, output_dir)
        
        return exported_path
    
    def _create_deployment_package(self, model_path: str, platform: str, output_dir: str):
        """Create deployment package with model and metadata"""
        try:
            import json
            
            # Create metadata
            metadata = {
                'model_path': os.path.basename(model_path),
                'platform': platform,
                'export_timestamp': time.time(),
                'opset_version': self.opset_version,
                'optimized': self.optimize,
                'quantized': self.quantize,
                'export_stats': self.export_stats,
                'deployment_notes': self._get_deployment_notes(platform)
            }
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"{platform}_deployment_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Deployment package created at {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to create deployment package: {str(e)}")
    
    def _get_deployment_notes(self, platform: str) -> Dict[str, str]:
        """Get platform-specific deployment notes"""
        notes = {
            'raspberry_pi': {
                'installation': 'pip install onnxruntime',
                'performance': 'Expected inference time: 10-50ms',
                'memory': 'Requires ~100MB RAM',
                'optimization': 'Model quantized for ARM CPU'
            },
            'ai_hat': {
                'installation': 'Use Hailo runtime',
                'performance': 'Expected inference time: 1-5ms',
                'memory': 'Uses dedicated AI accelerator',
                'optimization': 'Optimized for Hailo-8 NPU'
            },
            'generic': {
                'installation': 'pip install onnxruntime',
                'performance': 'Varies by hardware',
                'memory': 'Optimized for edge deployment',
                'optimization': 'General edge optimizations applied'
            }
        }
        
        return notes.get(platform, notes['generic'])
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of export process"""
        return {
            'model_class': self.model.__class__.__name__,
            'opset_version': self.opset_version,
            'optimizations_applied': self.optimize,
            'quantization_applied': self.quantize,
            'export_stats': self.export_stats
        }

