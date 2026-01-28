from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routes import video

app = FastAPI(
    title = settings.APP_NAME,
    description="AI-powered football match analysis API using computer vision",
    version = settings.VERSION,
    debug = settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video.router)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "upload_video": "POST /video/upload",
            "check_status": "GET /video/status/{task_id}",
            "get_result": "GET /video/result/{task_id}",
            "download_file": "GET /video/download/{task_id}/{filename}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_exists": os.path.exists(settings.MODEL_PATH),
        "upload_dir_writable": os.access(settings.UPLOAD_DIR, os.W_OK),
        "output_dir_writable": os.access(settings.OUTPUT_DIR, os.W_OK)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=settings.DEBUG
    )    

