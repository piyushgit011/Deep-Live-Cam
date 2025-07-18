import os
import logging

logger = logging.getLogger(__name__)

def optimize_gpu_settings():
    """Optimize GPU settings for better performance"""
    try:
        # CUDA settings
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        
        # Memory optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        logger.info("🚀 GPU optimization settings applied")
        
    except Exception as e:
        logger.error(f"⚠️ Error applying GPU optimizations: {e}")

# Call this in main.py startup
if __name__ == "__main__":
    optimize_gpu_settings()
