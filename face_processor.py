import cv2
import numpy as np
import os
import sys
import logging
import asyncio
import threading
import queue
import time
from typing import Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import GPU optimizations
from gpu_config import optimize_gpu_settings_rtx4090

logger = logging.getLogger(__name__)

class HighPerformanceFaceSwapProcessor:
    """Optimized face swapping processor for RTX 4090"""
    
    def __init__(self):
        self.source_face = None
        self.face_analyser = None
        self.face_swapper = None
        self.initialized = False
        self.get_one_face = None
        self.swap_face = None
        
        # Performance optimizations
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.processing_thread = None
        self.is_processing = False
        
        # Threading for async processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.process_lock = threading.Lock()
        
        # Performance tracking
        self.frame_count = 0
        self.process_times = []
        self.last_fps_time = time.time()
        
        # Apply RTX 4090 optimizations
        optimize_gpu_settings_rtx4090()
        
    def detect_execution_providers(self):
        """Detect and configure optimal execution providers for RTX 4090"""
        try:
            import onnxruntime
            available_providers = onnxruntime.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")
            
            # RTX 4090 optimized provider settings
            if 'CUDAExecutionProvider' in available_providers:
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'gpu_mem_limit': 16 * 1024 * 1024 * 1024,  # 16GB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                    'cudnn_conv_use_max_workspace': True,
                    'cudnn_conv1d_pad_to_nc1d': True,
                }
                logger.info("🚀 Using optimized CUDA execution provider for RTX 4090")
                return [('CUDAExecutionProvider', cuda_options)]
            
            # Fallback providers
            preferred_providers = [
                'ROCMExecutionProvider', 
                'CoreMLExecutionProvider',
                'DirectMLExecutionProvider',
                'OpenVINOExecutionProvider',
                'CPUExecutionProvider'
            ]
            
            for provider in preferred_providers:
                if provider in available_providers:
                    logger.info(f"Selected execution provider: {provider}")
                    return [provider]
            
            logger.warning("No GPU providers available, using CPU")
            return ['CPUExecutionProvider']
            
        except Exception as e:
            logger.error(f"Error detecting execution providers: {e}")
            return ['CPUExecutionProvider']
        
    async def initialize(self):
        """Initialize face processing models with RTX 4090 optimizations"""
        try:
            logger.info("Initializing high-performance face processing models...")
            
            # Import modules after adding to path
            import modules.globals
            from modules.face_analyser import get_face_analyser, get_one_face
            from modules.processors.frame.face_swapper import get_face_swapper, swap_face
            
            # Store module references
            self.get_one_face = get_one_face
            self.swap_face = swap_face
            
            # Configure execution providers for RTX 4090
            execution_providers = self.detect_execution_providers()
            modules.globals.execution_providers = execution_providers
            
            # RTX 4090 optimized settings
            modules.globals.execution_threads = 12  # RTX 4090 has many cores
            modules.globals.max_memory = 20  # Use more memory for better performance
            
            # Initialize models with optimizations
            logger.info("Initializing face analysis model...")
            self.face_analyser = get_face_analyser()
            
            logger.info("Initializing face swapper model...")
            self.face_swapper = get_face_swapper()
            
            # Warm up the models
            await self._warm_up_models()
            
            self.initialized = True
            logger.info("✅ High-performance face processing models initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing face processing models: {str(e)}")
            logger.error(f"Exception details: ", exc_info=True)
            self.initialized = False
    
    async def _warm_up_models(self):
        """Warm up models with dummy data for optimal performance"""
        try:
            logger.info("🔥 Warming up models...")
            # Create dummy frame for warm-up
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Warm up face analyzer
            _ = self.get_one_face(dummy_frame)
            
            logger.info("✅ Models warmed up successfully")
        except Exception as e:
            logger.warning(f"⚠️ Model warm-up failed: {e}")
    
    async def set_source_face(self, image: np.ndarray) -> bool:
        """Set source face from uploaded image"""
        try:
            if not self.initialized:
                logger.warning("Face processor not initialized, attempting to initialize...")
                await self.initialize()
                
            if not self.initialized:
                logger.error("Cannot set source face: processor not initialized")
                return False
            
            # Extract face from image
            with self.process_lock:
                face = self.get_one_face(image)
            
            if face is not None:
                self.source_face = face
                logger.info("✅ Source face set successfully")
                return True
            else:
                logger.warning("⚠️ No face detected in source image")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error setting source face: {str(e)}")
            return False
    
    def process_frame_sync(self, frame: np.ndarray) -> np.ndarray:
        """Synchronous frame processing optimized for RTX 4090"""
        if not self.initialized or self.source_face is None:
            return frame
        
        start_time = time.time()
        
        try:
            # Get target face from frame
            target_face = self.get_one_face(frame)
            
            if target_face is not None:
                # Perform face swap
                swapped_frame = self.swap_face(self.source_face, target_face, frame)
                
                # Track performance
                process_time = (time.time() - start_time) * 1000
                self.process_times.append(process_time)
                if len(self.process_times) > 30:  # Keep last 30 measurements
                    self.process_times.pop(0)
                
                self.frame_count += 1
                
                # Log FPS every 60 frames
                if self.frame_count % 60 == 0:
                    current_time = time.time()
                    time_diff = current_time - self.last_fps_time
                    fps = 60 / time_diff if time_diff > 0 else 0
                    
                    if self.process_times:
                        avg_process_time = sum(self.process_times) / len(self.process_times)
                        logger.info(f"🚀 Processing FPS: {fps:.1f}, Avg time: {avg_process_time:.1f}ms")
                    
                    self.last_fps_time = current_time
                
                return swapped_frame
            else:
                # No face detected in frame, return original
                return frame
                
        except Exception as e:
            logger.error(f"❌ Error processing frame: {str(e)}")
            return frame
    
    async def process_frame_async(self, frame: np.ndarray) -> np.ndarray:
        """Asynchronous frame processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.process_frame_sync, frame)
    
    def has_source_face(self) -> bool:
        """Check if source face is loaded"""
        return self.source_face is not None and self.initialized
    
    def clear_source_face(self):
        """Clear stored source face"""
        self.source_face = None
        logger.info("🗑️ Source face cleared")
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        if not self.process_times:
            return {"avg_process_time": 0, "fps": 0, "frames_processed": self.frame_count}
        
        avg_time = sum(self.process_times) / len(self.process_times)
        estimated_fps = 1000 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_process_time": round(avg_time, 2),
            "fps": round(estimated_fps, 1),
            "frames_processed": self.frame_count
        }