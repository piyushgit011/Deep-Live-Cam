import os
import logging
import torch

logger = logging.getLogger(__name__)

def optimize_gpu_settings_rtx4090():
    """Optimize GPU settings specifically for RTX 4090"""
    try:
        # RTX 4090 specific CUDA settings
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        
        # High-performance memory settings for RTX 4090
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,roundup_power2_divisions:16'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        
        # Enable Tensor Core optimizations
        os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
        
        # Memory pool settings for 24GB VRAM
        os.environ['PYTORCH_CUDA_MEMORY_POOL_INIT'] = '8192'  # 8GB initial pool
        
        # CUDNN optimizations
        os.environ['CUDNN_BENCHMARK'] = '1'
        os.environ['CUDNN_DETERMINISTIC'] = '0'
        
        logger.info("🚀 RTX 4090 optimization settings applied")
        
        # PyTorch specific optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction (use ~16GB out of 24GB)
            torch.cuda.set_per_process_memory_fraction(0.7)
            
            logger.info(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
    except Exception as e:
        logger.error(f"⚠️ Error applying RTX 4090 optimizations: {e}")

if __name__ == "__main__":
    optimize_gpu_settings_rtx4090()