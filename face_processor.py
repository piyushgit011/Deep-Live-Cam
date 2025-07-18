import cv2
import numpy as np
import os
import sys
import logging
import asyncio
from typing import Optional, Any
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class FaceSwapProcessor:
    """Face swapping processor for real-time video streams"""
    
    def __init__(self):
        self.source_face = None
        self.face_analyser = None
        self.face_swapper = None
        self.initialized = False
        self.get_one_face = None
        self.swap_face = None
        
    def detect_execution_providers(self):
        """Detect available execution providers"""
        try:
            import onnxruntime
            available_providers = onnxruntime.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")
            
            # Priority order for execution providers
            preferred_providers = [
                'CUDAExecutionProvider',
                'ROCMExecutionProvider', 
                'CoreMLExecutionProvider',
                'DirectMLExecutionProvider',
                'OpenVINOExecutionProvider',
                'CPUExecutionProvider'
            ]
            
            # Select the best available provider
            for provider in preferred_providers:
                if provider in available_providers:
                    logger.info(f"Selected execution provider: {provider}")
                    return [provider]
            
            # Fallback to CPU
            logger.warning("No GPU providers available, using CPU")
            return ['CPUExecutionProvider']
            
        except Exception as e:
            logger.error(f"Error detecting execution providers: {e}")
            return ['CPUExecutionProvider']
        
    async def initialize(self):
        """Initialize face processing models"""
        try:
            logger.info("Initializing face processing models...")
            
            # Import modules after adding to path
            import modules.globals
            from modules.face_analyser import get_face_analyser, get_one_face
            from modules.processors.frame.face_swapper import get_face_swapper, swap_face
            
            # Store module references
            self.get_one_face = get_one_face
            self.swap_face = swap_face
            
            # Detect and set execution providers
            execution_providers = self.detect_execution_providers()
            modules.globals.execution_providers = execution_providers
            
            # Set optimal configuration based on provider
            if 'CUDAExecutionProvider' in execution_providers:
                logger.info("🚀 Using CUDA GPU acceleration")
                modules.globals.execution_threads = 8
                modules.globals.max_memory = 16
            elif 'ROCMExecutionProvider' in execution_providers:
                logger.info("🚀 Using ROCm GPU acceleration")
                modules.globals.execution_threads = 8
                modules.globals.max_memory = 16
            else:
                logger.info("🔄 Using CPU execution")
                modules.globals.execution_threads = 4
                modules.globals.max_memory = 8
            
            # Initialize models
            logger.info("Initializing face analysis model...")
            self.face_analyser = get_face_analyser()
            
            logger.info("Initializing face swapper model...")
            self.face_swapper = get_face_swapper()
            
            self.initialized = True
            logger.info("✅ Face processing models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing face processing models: {str(e)}")
            logger.error(f"Exception details: ", exc_info=True)
            self.initialized = False
    
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
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with face swapping"""
        try:
            if not self.initialized or self.source_face is None:
                return frame
            
            # Get target face from frame
            target_face = self.get_one_face(frame)
            
            if target_face is not None:
                # Perform face swap
                swapped_frame = self.swap_face(self.source_face, target_face, frame)
                return swapped_frame
            else:
                # No face detected in frame, return original
                return frame
                
        except Exception as e:
            logger.error(f"❌ Error processing frame: {str(e)}")
            return frame
    
    def has_source_face(self) -> bool:
        """Check if source face is loaded"""
        return self.source_face is not None and self.initialized
    
    def clear_source_face(self):
        """Clear stored source face"""
        self.source_face = None
        logger.info("🗑️ Source face cleared")