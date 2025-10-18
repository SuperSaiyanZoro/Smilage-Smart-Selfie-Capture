from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# Initialize FastAPI app
app = FastAPI(title="Smilage - Smart Selfie API")

# Configure CORS (allows frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create captured_images directory if it doesn't exist
CAPTURED_IMAGES_DIR = "captured_images"
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)

# Serve captured images as static files
app.mount("/captured_images", StaticFiles(directory=CAPTURED_IMAGES_DIR), name="captured_images")


# Root endpoint - Health check
@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running
    """
    return {
        "message": "Smilage Smart Selfie API is running!",
        "status": "active",
        "version": "1.0.0"
    }


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "service": "Smilage Smart Selfie"
    }


# System info endpoint
@app.get("/api/system-info")
async def system_info():
    """
    Returns basic system information
    """
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "cpu_usage": f"{cpu_percent}%",
        "memory_usage": f"{memory.percent}%",
        "memory_available": f"{memory.available / (1024**3):.2f} GB"
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Smilage Smart Selfie Backend Server...")
    print("📍 Server will run at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
