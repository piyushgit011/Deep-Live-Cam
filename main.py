from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import cv2
import numpy as np
import asyncio
import json
import uuid
import logging
import base64
import time
from pathlib import Path

# Import optimized face processing modules
from face_processor import HighPerformanceFaceSwapProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
face_processor = HighPerformanceFaceSwapProcessor()
active_websockets = set()

# Performance tracking
frame_stats = {
    "frames_received": 0,
    "frames_processed": 0,
    "start_time": time.time(),
    "last_fps_log": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler"""
    # Startup
    logger.info("🚀 Starting High-Performance Deep Live Cam WebSocket API...")
    
    # Create required directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Initialize face processor
    try:
        await face_processor.initialize()
        logger.info("✅ High-performance face processor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize face processor: {str(e)}")
    
    logger.info("🎭 High-Performance Deep Live Cam WebSocket API ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Deep Live Cam WebSocket API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="High-Performance Deep Live Cam WebSocket API", 
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    html_file = static_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    else:
        return HTMLResponse(content="<h1>Please create static/index.html</h1>")

@app.post("/upload-face")
async def upload_face(file: UploadFile = File(...)):
    """Upload source face image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process and store the face
        success = await face_processor.set_source_face(img)
        
        if not success:
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")
        
        return JSONResponse(content={
            "message": "Face uploaded successfully", 
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Error uploading face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    """High-performance WebSocket endpoint for video streaming"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    logger.info(f"📹 WebSocket connection established. Active connections: {len(active_websockets)}")
    
    # Connection-specific stats
    connection_stats = {
        "frames_received": 0,
        "frames_processed": 0,
        "start_time": time.time(),
        "last_log_time": time.time()
    }
    
    try:
        while True:
            # Receive video frame data
            data = await websocket.receive_text()
            
            try:
                # Parse JSON data
                frame_data = json.loads(data)
                
                if frame_data.get("type") == "video_frame":
                    connection_stats["frames_received"] += 1
                    frame_stats["frames_received"] += 1
                    
                    # Process the video frame asynchronously
                    response = await process_video_frame_optimized(frame_data, connection_stats)
                    await websocket.send_text(json.dumps(response))
                    
                    connection_stats["frames_processed"] += 1
                    frame_stats["frames_processed"] += 1
                    
                    # Log performance every 120 frames
                    if connection_stats["frames_processed"] % 120 == 0:
                        await log_performance_stats(connection_stats)
                    
                elif frame_data.get("type") == "ping":
                    # Respond to ping with pong and performance stats
                    stats = face_processor.get_performance_stats()
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "stats": stats
                    }))
                    
            except json.JSONDecodeError:
                logger.error("❌ Invalid JSON received")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"❌ Error processing frame: {str(e)}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": str(e)
                }))
                
    except Exception as e:
        logger.error(f"❌ WebSocket error: {str(e)}")
    finally:
        active_websockets.discard(websocket)
        logger.info(f"📹 WebSocket connection closed. Active connections: {len(active_websockets)}")

async def process_video_frame_optimized(frame_data, connection_stats):
    """Optimized video frame processing"""
    process_start = time.time()
    
    try:
        # Decode base64 image
        image_data = frame_data.get("data")
        if not image_data:
            return {"type": "error", "message": "No image data provided"}
        
        # Decode base64 to bytes (optimized)
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"❌ Base64 decode error: {str(e)}")
            return {"type": "error", "message": f"Base64 decode failed: {str(e)}"}
        
        # Convert to numpy array (optimized)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("❌ Failed to decode image from bytes")
            return {"type": "error", "message": "Failed to decode image"}
        
        decode_time = (time.time() - process_start) * 1000
        
        # Process frame with face swapping if source face is loaded
        if face_processor.has_source_face():
            # Use async processing for better performance
            processed_frame = await face_processor.process_frame_async(frame)
        else:
            processed_frame = frame
        
        process_time = (time.time() - process_start) * 1000
        
        # Encode processed frame back to base64 (optimized quality)
        encode_start = time.time()
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_data = base64.b64encode(buffer).decode('utf-8')
        encode_time = (time.time() - encode_start) * 1000
        
        total_time = (time.time() - process_start) * 1000
        
        return {
            "type": "processed_frame",
            "data": processed_data,
            "timestamp": frame_data.get("timestamp", 0),
            "performance": {
                "decode_time": round(decode_time, 2),
                "process_time": round(process_time, 2),
                "encode_time": round(encode_time, 2),
                "total_time": round(total_time, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing video frame: {str(e)}")
        return {"type": "error", "message": str(e)}

async def log_performance_stats(connection_stats):
    """Log performance statistics"""
    current_time = time.time()
    time_diff = current_time - connection_stats["start_time"]
    
    if time_diff > 0:
        avg_fps = connection_stats["frames_processed"] / time_diff
        face_stats = face_processor.get_performance_stats()
        
        logger.info(f"🚀 Performance Stats:")
        logger.info(f"   Connection FPS: {avg_fps:.1f}")
        logger.info(f"   Processing FPS: {face_stats.get('fps', 0)}")
        logger.info(f"   Avg Process Time: {face_stats.get('avg_process_time', 0)}ms")
        logger.info(f"   Total Frames: {connection_stats['frames_processed']}")

@app.get("/status")
async def get_status():
    """Get application status with performance metrics"""
    stats = face_processor.get_performance_stats()
    
    return JSONResponse(content={
        "status": "running",
        "face_loaded": face_processor.has_source_face(),
        "active_websocket_connections": len(active_websockets),
        "websocket_enabled": True,
        "performance": stats,
        "global_stats": frame_stats
    })

@app.get("/performance")
async def get_performance():
    """Get detailed performance metrics"""
    return JSONResponse(content={
        "face_processor": face_processor.get_performance_stats(),
        "global_stats": frame_stats,
        "active_connections": len(active_websockets)
    })

@app.post("/clear-face")
async def clear_face():
    """Clear uploaded face"""
    face_processor.clear_source_face()
    return JSONResponse(content={"message": "Face cleared successfully"})

if __name__ == "__main__":
    # Run with optimized settings for RTX 4090
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        loop="asyncio",
        workers=1  # Single worker for GPU processing
    )