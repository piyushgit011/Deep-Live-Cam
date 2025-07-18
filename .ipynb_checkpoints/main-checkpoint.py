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
from pathlib import Path

# Import face processing modules
from face_processor import FaceSwapProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
face_processor = FaceSwapProcessor()
active_websockets = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler"""
    # Startup
    logger.info("🚀 Starting Deep Live Cam WebSocket API...")
    
    # Create required directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Initialize face processor
    try:
        await face_processor.initialize()
        logger.info("✅ Face processor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize face processor: {str(e)}")
    
    logger.info("🎭 Deep Live Cam WebSocket API ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Deep Live Cam WebSocket API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Deep Live Cam WebSocket API", 
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
    """WebSocket endpoint for video streaming"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    logger.info(f"📹 WebSocket connection established. Active connections: {len(active_websockets)}")
    
    frame_count = 0
    
    try:
        while True:
            # Receive video frame data
            data = await websocket.receive_text()
            
            try:
                # Parse JSON data
                frame_data = json.loads(data)
                
                if frame_data.get("type") == "video_frame":
                    frame_count += 1
                    logger.info(f"🎬 Processing frame {frame_count}")
                    
                    # Process the video frame
                    response = await process_video_frame(frame_data)
                    await websocket.send_text(json.dumps(response))
                    
                elif frame_data.get("type") == "ping":
                    # Respond to ping with pong
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
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

async def process_video_frame(frame_data):
    """Process a single video frame"""
    try:
        # Decode base64 image
        image_data = frame_data.get("data")
        if not image_data:
            return {"type": "error", "message": "No image data provided"}
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 to bytes
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"❌ Base64 decode error: {str(e)}")
            return {"type": "error", "message": f"Base64 decode failed: {str(e)}"}
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("❌ Failed to decode image from bytes")
            return {"type": "error", "message": "Failed to decode image"}
        
        logger.info(f"📷 Frame decoded: {frame.shape}")
        
        # Process frame with face swapping if source face is loaded
        if face_processor.has_source_face():
            logger.info("🎭 Processing frame with face swap")
            processed_frame = face_processor.process_frame(frame)
        else:
            logger.warning("⚠️ No source face loaded, returning original frame")
            processed_frame = frame
        
        # Encode processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed_data = base64.b64encode(buffer).decode('utf-8')
        
        logger.info("✅ Frame processed successfully")
        
        return {
            "type": "processed_frame",
            "data": processed_data,
            "timestamp": frame_data.get("timestamp", 0)
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing video frame: {str(e)}")
        return {"type": "error", "message": str(e)}

@app.get("/status")
async def get_status():
    """Get application status"""
    return JSONResponse(content={
        "status": "running",
        "face_loaded": face_processor.has_source_face(),
        "active_websocket_connections": len(active_websockets),
        "execution_provider": getattr(face_processor, 'execution_provider', 'Unknown'),
        "websocket_enabled": True
    })

@app.post("/clear-face")
async def clear_face():
    """Clear uploaded face"""
    face_processor.clear_source_face()
    return JSONResponse(content={"message": "Face cleared successfully"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")