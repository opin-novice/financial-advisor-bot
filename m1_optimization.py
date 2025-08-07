#!/usr/bin/env python3
"""
M1 MacBook Air Optimization Script for RAG System
Optimizes system settings and environment for 8GB RAM M1 MacBook Air
"""

import os
import sys
import psutil
import torch
import gc
from typing import Dict, Any

class M1Optimizer:
    def __init__(self):
        self.system_info = self._get_system_info()
        self.optimizations_applied = []

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "total_memory": psutil.virtual_memory().total / (1024**3),  # GB
            "available_memory": psutil.virtual_memory().available / (1024**3),  # GB
            "cpu_count": psutil.cpu_count(),
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "cuda_available": torch.cuda.is_available()
        }

    def apply_m1_optimizations(self):
        """Apply M1-specific optimizations"""
        print("🚀 Applying M1 MacBook Air optimizations...")
        
        # 1. Environment Variables
        self._set_environment_variables()
        
        # 2. PyTorch Optimizations
        self._optimize_pytorch()
        
        # 3. Memory Management
        self._optimize_memory_management()
        
        # 4. Threading Optimizations
        self._optimize_threading()
        
        # 5. Model Loading Optimizations
        self._optimize_model_loading()
        
        print(f"✅ Applied {len(self.optimizations_applied)} optimizations")
        self._print_optimization_summary()

    def _set_environment_variables(self):
        """Set M1-optimized environment variables"""
        env_vars = {
            # Tokenizers optimization
            "TOKENIZERS_PARALLELISM": "false",
            
            # OpenMP optimizations for M1
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "VECLIB_MAXIMUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            
            # Memory optimizations
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            
            # Hugging Face optimizations
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            
            # Disable unnecessary features
            "WANDB_DISABLED": "true",
            "DISABLE_MLFLOW_INTEGRATION": "true"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            
        self.optimizations_applied.append("Environment variables configured")

    def _optimize_pytorch(self):
        """Optimize PyTorch for M1"""
        # Set number of threads
        torch.set_num_threads(4)
        
        # Enable MPS if available
        if self.system_info["mps_available"]:
            print("🔥 MPS (Metal Performance Shaders) available - enabling acceleration")
            # MPS optimizations will be handled in model initialization
        
        # Memory management
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')  # Trade precision for speed
            
        self.optimizations_applied.append("PyTorch optimized for M1")

    def _optimize_memory_management(self):
        """Optimize memory management for 8GB RAM"""
        # Force garbage collection
        gc.collect()
        
        # Set memory growth strategy
        if self.system_info["mps_available"]:
            # MPS memory management
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        self.optimizations_applied.append("Memory management optimized")

    def _optimize_threading(self):
        """Optimize threading for M1 performance cores"""
        # M1 has 4 performance cores + 4 efficiency cores
        # Limit to performance cores for CPU-intensive tasks
        optimal_threads = min(4, psutil.cpu_count())
        
        # Set thread limits
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(2)
        
        self.optimizations_applied.append(f"Threading optimized ({optimal_threads} threads)")

    def _optimize_model_loading(self):
        """Optimize model loading strategies"""
        # These will be applied during model initialization
        model_optimizations = {
            "use_fast_tokenizer": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": "float16",  # Use half precision when possible
            "device_map": "auto"
        }
        
        # Store optimizations for later use
        self.model_optimizations = model_optimizations
        self.optimizations_applied.append("Model loading strategies configured")

    def get_optimal_batch_size(self, model_type: str = "embedding") -> int:
        """Calculate optimal batch size based on available memory"""
        available_gb = self.system_info["available_memory"]
        
        if model_type == "embedding":
            # Conservative batch sizes for embedding models
            if available_gb > 6:
                return 32
            elif available_gb > 4:
                return 16
            else:
                return 8
        elif model_type == "cross_encoder":
            # Smaller batches for cross-encoder
            if available_gb > 6:
                return 16
            elif available_gb > 4:
                return 8
            else:
                return 4
        
        return 4  # Conservative default

    def get_optimal_device(self) -> str:
        """Get optimal device for model inference"""
        if self.system_info["mps_available"]:
            return "mps"
        elif self.system_info["cuda_available"]:
            return "cuda"
        else:
            return "cpu"

    def monitor_memory_usage(self):
        """Monitor current memory usage"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percentage": memory.percent
        }

    def _print_optimization_summary(self):
        """Print optimization summary"""
        print("\n📊 System Information:")
        print(f"   Platform: {self.system_info['platform']}")
        print(f"   Total Memory: {self.system_info['total_memory']:.1f} GB")
        print(f"   Available Memory: {self.system_info['available_memory']:.1f} GB")
        print(f"   CPU Cores: {self.system_info['cpu_count']}")
        print(f"   PyTorch Version: {self.system_info['torch_version']}")
        print(f"   MPS Available: {self.system_info['mps_available']}")
        
        print("\n✅ Optimizations Applied:")
        for opt in self.optimizations_applied:
            print(f"   • {opt}")
        
        print(f"\n🎯 Recommended Settings:")
        print(f"   • Device: {self.get_optimal_device()}")
        print(f"   • Embedding Batch Size: {self.get_optimal_batch_size('embedding')}")
        print(f"   • Cross-Encoder Batch Size: {self.get_optimal_batch_size('cross_encoder')}")

def optimize_for_m1():
    """Main optimization function"""
    optimizer = M1Optimizer()
    optimizer.apply_m1_optimizations()
    return optimizer

if __name__ == "__main__":
    # Run optimizations
    optimizer = optimize_for_m1()
    
    # Monitor memory
    memory_info = optimizer.monitor_memory_usage()
    print(f"\n💾 Current Memory Usage: {memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB ({memory_info['percentage']:.1f}%)")
    
    if memory_info['percentage'] > 80:
        print("⚠️  Warning: High memory usage detected. Consider closing other applications.")
    else:
        print("✅ Memory usage is optimal for RAG operations.")
