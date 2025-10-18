from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import base64
import json
import os
import asyncio
from datetime import datetime
from typing import List
import psutil
from utils.video_processor import VideoProcessor

# Initialize FastAPI app
app = FastAPI(title="Smilage - Smart Selfie API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
CAPTURED_IMAGES_DIR = "captured_images"
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)

# Mount static files
app.mount("/captured_images", StaticFiles(directory=CAPTURED_IMAGES_DIR), name="captured_images")

# Global video processor
video_processor = None

# Active camera state
camera_active = False
camera = None


def get_video_processor():
    """Get or create video processor instance"""
    global video_processor
    if video_processor is None:
        video_processor = VideoProcessor()
    return video_processor


# ==================== ROOT & HEALTH ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smilage Smart Selfie API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "websocket": "/ws",
            "gallery": "/api/gallery"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Smilage Smart Selfie",
        "camera_active": camera_active
    }


# ==================== SYSTEM INFO ENDPOINT ====================

@app.get("/api/system-info")
async def system_info():
    """Get system performance metrics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    return {
        "cpu_usage": f"{cpu_percent}%",
        "cpu_percent": cpu_percent,
        "memory_usage": f"{memory.percent}%",
        "memory_percent": memory.percent,
        "memory_available": f"{memory.available / (1024**3):.2f} GB",
        "memory_total": f"{memory.total / (1024**3):.2f} GB"
    }


# ==================== SETTINGS ENDPOINTS ====================

@app.post("/api/settings/smile-threshold")
async def update_smile_threshold(data: dict):
    """Update smile detection threshold"""
    threshold = data.get("threshold", 0.5)
    processor = get_video_processor()
    new_threshold = processor.set_smile_threshold(threshold)
    
    return {
        "success": True,
        "threshold": new_threshold
    }


@app.get("/api/settings")
async def get_settings():
    """Get current settings"""
    processor = get_video_processor()
    
    return {
        "smile_threshold": processor.smile_threshold,
        "capture_dir": processor.capture_dir
    }


# ==================== GALLERY ENDPOINTS ====================

@app.get("/api/gallery")
async def get_gallery():
    """Get list of captured images"""
    try:
        files = []
        for filename in os.listdir(CAPTURED_IMAGES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(CAPTURED_IMAGES_DIR, filename)
                stat = os.stat(filepath)
                
                files.append({
                    "filename": filename,
                    "url": f"/captured_images/{filename}",
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x['created'], reverse=True)
        
        return {
            "success": True,
            "count": len(files),
            "images": files
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "images": []
        }


@app.delete("/api/gallery/{filename}")
async def delete_image(filename: str):
    """Delete a captured image"""
    try:
        filepath = os.path.join(CAPTURED_IMAGES_DIR, filename)
        
        if not os.path.exists(filepath):
            return {
                "success": False,
                "error": "File not found"
            }
        
        os.remove(filepath)
        
        return {
            "success": True,
            "message": f"Deleted {filename}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.delete("/api/gallery")
async def clear_gallery():
    """Delete all captured images"""
    try:
        count = 0
        for filename in os.listdir(CAPTURED_IMAGES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(CAPTURED_IMAGES_DIR, filename)
                os.remove(filepath)
                count += 1
        
        return {
            "success": True,
            "message": f"Deleted {count} images"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/gallery/download/{filename}")
async def download_image(filename: str):
    """Download a captured image"""
    filepath = os.path.join(CAPTURED_IMAGES_DIR, filename)
    
    if not os.path.exists(filepath):
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )
    
    return FileResponse(
        filepath,
        media_type="image/jpeg",
        filename=filename
    )


# ==================== CAPTURE ENDPOINT ====================

@app.post("/api/capture")
async def manual_capture():
    """Manually capture current frame"""
    global camera
    
    if camera is None or not camera.isOpened():
        return {
            "success": False,
            "error": "Camera not active"
        }
    
    try:
        ret, frame = camera.read()
        
        if not ret:
            return {
                "success": False,
                "error": "Failed to capture frame"
            }
        
        processor = get_video_processor()
        predictions = processor.process_frame(frame)
        capture_info = processor.capture_selfie(frame, predictions)
        
        return {
            "success": True,
            "image": capture_info,
            "predictions": predictions
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ==================== WEBSOCKET ENDPOINT ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    global camera_active, camera
    
    await websocket.accept()
    print("📡 WebSocket client connected")
    
    # Initialize video processor
    processor = get_video_processor()
    
    # Open camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    
    if not camera.isOpened():
        await websocket.send_json({
            "type": "error",
            "message": "Failed to open camera"
        })
        await websocket.close()
        return
    
    camera_active = True
    auto_capture_enabled = False
    frame_count = 0
    
    try:
        while True:
            # Check for incoming messages
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=0.001
                )
                
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "stop":
                    print("🛑 Stop signal received")
                    break
                elif msg_type == "capture":
                    # Manual capture
                    ret, frame = camera.read()
                    if ret:
                        predictions = processor.process_frame(frame)
                        capture_info = processor.capture_selfie(frame, predictions)
                        
                        await websocket.send_json({
                            "type": "capture_success",
                            "image": capture_info
                        })
                elif msg_type == "auto_capture":
                    auto_capture_enabled = data.get("enabled", False)
                    print(f"🤖 Auto-capture: {auto_capture_enabled}")
                elif msg_type == "settings":
                    if "smile_threshold" in data:
                        processor.set_smile_threshold(data["smile_threshold"])
                        
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass
            
            # Read frame from camera
            ret, frame = camera.read()
            
            if not ret:
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to read frame"
                })
                break
            
            frame_count += 1
            
            # Process frame
            predictions = processor.process_frame(frame)
            
            # Auto-capture on smile
            if auto_capture_enabled and len(predictions["faces"]) > 0:
                for face in predictions["faces"]:
                    if face["is_smiling"] and face["is_clear"]:
                        capture_info = processor.capture_selfie(frame, predictions)
                        
                        await websocket.send_json({
                            "type": "auto_capture",
                            "image": capture_info
                        })
                        
                        # Disable auto-capture temporarily
                        auto_capture_enabled = False
                        break
            
            # Draw predictions on frame
            annotated_frame = processor.draw_predictions(frame.copy(), predictions)
            
            # Encode frame
            frame_base64 = processor.encode_frame_to_base64(annotated_frame)
            
            # Send frame and predictions
            await websocket.send_json({
                "type": "frame",
                "frame": frame_base64,
                "predictions": predictions,
                "frame_number": frame_count
            })
            
            # Small delay to control frame rate
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except WebSocketDisconnect:
        print("📡 WebSocket client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None
        print("🎥 Camera released")


# ==================== STARTUP & SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("="*60)
    print("🚀 Smilage Smart Selfie Backend Starting...")
    print("="*60)
    print("📍 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws")
    print("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera
    if camera is not None:
        camera.release()
    print("👋 Smilage backend shut down")


# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
