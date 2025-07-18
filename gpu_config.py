import os
import logging
import torch

logger = logging.getLogger(__name__)

def optimize_gpu_settings_rtx4090():
    """Optimize GPU settings specifically for RTX 4090 in RunPod - Fixed Version"""
    try:
        # Set environment for CUDA 12 compatibility
        os.environ['CUDA_HOME'] = '/usr/local/cuda'
        os.environ['CUDA_ROOT'] = '/usr/local/cuda'
        
        # Don't override RunPod's GPU allocation
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Memory and performance optimizations
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        
        # Conservative memory settings for container environment
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,roundup_power2_divisions:16'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        
        # Enable optimizations
        os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
        os.environ['CUDNN_BENCHMARK'] = '1'
        os.environ['CUDNN_DETERMINISTIC'] = '0'
        
        logger.info("🚀 GPU optimization settings applied (CUDA 12 compatible)")
        
        # PyTorch specific optimizations with error handling
        try:
            # Force reinitialize CUDA context
            if hasattr(torch.cuda, '_lazy_init'):
                torch.cuda._lazy_init()
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"🎯 Found {device_count} CUDA device(s)")
                
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    
                    logger.info(f"🎯 GPU: {device_name}")
                    logger.info(f"💾 VRAM: {total_memory / 1024**3:.1f}GB")
                    
                    # Configure PyTorch optimizations
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    
                    # Conservative memory fraction for containers
                    memory_fraction = 0.7 if total_memory > 20 * 1024**3 else 0.8
                    torch.cuda.set_per_process_memory_fraction(memory_fraction)
                    logger.info(f"💾 Set memory fraction: {memory_fraction}")
                    
                    # Clear any existing cache
                    torch.cuda.empty_cache()
                    logger.info("🧹 Cleared CUDA cache")
                    
                    return True
                else:
                    logger.warning("⚠️ CUDA available but no devices found")
                    return False
            else:
                logger.warning("⚠️ CUDA not available - falling back to CPU")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ PyTorch CUDA initialization failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"⚠️ Error applying GPU optimizations: {e}")
        return False

def check_gpu_status():
    """Check and return GPU status information"""
    status = {
        'cuda_available': False,
        'device_count': 0,
        'devices': [],
        'current_device': None,
        'memory_info': None,
        'libraries_loaded': {}
    }
    
    # Check CUDA libraries
    import ctypes
    try:
        ctypes.CDLL('libcudart.so.12')
        status['libraries_loaded']['cudart'] = True
    except:
        try:
            ctypes.CDLL('libcudart.so')
            status['libraries_loaded']['cudart'] = True
        except:
            status['libraries_loaded']['cudart'] = False
    
    try:
        ctypes.CDLL('libnvrtc.so.12')
        status['libraries_loaded']['nvrtc'] = True
    except:
        try:
            ctypes.CDLL('libnvrtc.so')
            status['libraries_loaded']['nvrtc'] = True
        except:
            status['libraries_loaded']['nvrtc'] = False
    
    # Check PyTorch CUDA
    try:
        import torch
        if hasattr(torch.cuda, '_lazy_init'):
            torch.cuda._lazy_init()
            
        status['cuda_available'] = torch.cuda.is_available()
        
        if status['cuda_available']:
            status['device_count'] = torch.cuda.device_count()
            status['current_device'] = torch.cuda.current_device()
            
            for i in range(status['device_count']):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'id': i,
                    'name': device_props.name,
                    'total_memory': device_props.total_memory,
                    'memory_gb': device_props.total_memory / 1024**3,
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                }
                status['devices'].append(device_info)
            
            # Get memory info for current device
            try:
                status['memory_info'] = {
                    'allocated': torch.cuda.memory_allocated(),
                    'cached': torch.cuda.memory_reserved(),
                    'max_allocated': torch.cuda.max_memory_allocated()
                }
            except:
                pass
    except Exception as e:
        status['error'] = str(e)
    
    return status

if __name__ == "__main__":
    # Test the configuration
    success = optimize_gpu_settings_rtx4090()
    status = check_gpu_status()
    
    print(f"GPU Configuration: {'✅ Success' if success else '❌ Failed'}")
    print(f"CUDA Available: {status['cuda_available']}")
    print(f"Device Count: {status['device_count']}")
    print(f"Libraries: {status['libraries_loaded']}")
    
    for device in status['devices']:
        print(f"GPU {device['id']}: {device['name']} ({device['memory_gb']:.1f}GB) - CC {device['compute_capability']}")
