import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import APIRouter , HTTPException, UploadFile , BackgroundTasks , File
from fastapi.responses import FileResponse
from backend.config import settings
from backend.schemas import (
    VideoUploadResponse, 
    TaskStatusResponse, 
    TaskStatus,
    AnalysisResult,
    Statistics
)
from backend.services.analysis import VideoAnalysisService

router = APIRouter(prefix='/video' , tags=['Video Analysis'])

tasks_db : Dict[str , Dict] = {}

def validate_video_file(file : UploadFile):
    """Validate uploaded video file"""
    if not file.filename :
        raise HTTPException(status_code=400 , detail="No filename Provided")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code= 400,
            detail=f"File type {file_extension} not allowed. Allowed: {list(settings.ALLOWED_EXTENSIONS)}"
        )

async def process_video_task(task_id :str , video_path : str , output_dir : str):
    """Process video analysis in the background"""

    tasks_db[task_id]["status"] = TaskStatus.PROCESSING
    tasks_db[task_id]["progress"] = 10
    tasks_db[task_id]["message"] = "Starting video analysis..."  

    try:
        service = VideoAnalysisService()

        tasks_db[task_id]["progress"] = 20
        tasks_db[task_id]["message"] = "Processing video frames..."

        result = service.process_video(video_path , output_dir , task_id)

        if result["success"]:
            tasks_db[task_id]["status"] = TaskStatus.COMPLETED
            tasks_db[task_id]["progress"] = 100
            tasks_db[task_id]["message"] = "Video analysis completed successfully"
            tasks_db[task_id]["completed_at"] = datetime.now()
            
            tasks_db[task_id]["result"] = {
                "video_url": f"/video/download/{task_id}/analyzed_video.avi",
                "statistics": result["statistics"],
                "tracks": result.get("tracks"),
                "team_ball_control": result.get("team_ball_control")
            }
        else:
            tasks_db[task_id]["status"] = TaskStatus.FAILED
            tasks_db[task_id]["message"] = "Analysis failed"
            tasks_db[task_id]["error"] = result.get("error", "Unknown error")
            tasks_db[task_id]["completed_at"] = datetime.now()

    except Exception as e:
        tasks_db[task_id]["status"] = TaskStatus.FAILED
        tasks_db[task_id]["message"] = "Processing error occurred"
        tasks_db[task_id]["error"] = str(e)
        tasks_db[task_id]["completed_at"] = datetime.now()


@router.post("/upload" , response_model = VideoUploadResponse)
async def upload_video(background_tasks : BackgroundTasks ,file : UploadFile  = File(...)):
    """Upload video for analysis"""
    validate_video_file(file)

    task_id = str(uuid.uuid4())
    upload_dir = settings.UPLOAD_DIR / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / file.filename

    try:
        with open(video_path , "wb") as buffer:
            content = await file.read()

            if len(content) > settings.MAX_FILE_SIZE:
                shutil.rmtree(upload_dir)
                raise HTTPException(status_code=400 , detail=f"File size exceeds limit. Max size: {settings.MAX_FILE_SIZE / (1024*1024)}MB")

            buffer.write(content)

    except Exception as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        

    output_dir = settings.OUTPUT_DIR / task_id
    output_dir.mkdir(parents = True , exist_ok = True)

    created_at = datetime.now()
    tasks_db[task_id] = {
        "task_id" : task_id,
        "status" : TaskStatus.PENDING,
        "progress"  : 0,
        "message": "Video uploaded, waiting to process",
        "created_at" : created_at,
        "completed_at" : None,
        "result" : None,
        "error" : None,
        "filename" : file.filename
    }


    background_tasks.add_task(process_video_task, task_id, str(video_path), str(output_dir))
        
    return VideoUploadResponse(
        task_id = task_id,
        status = TaskStatus.PENDING,
        message="Video uploaded successfully. Processing started.",
        created_at=created_at
    )
        

@router.get("/status/{task_id}" , response_model = TaskStatusResponse)
async def get_task_status(task_id : str):
    """
    Get processing status of a video analysis task
    
    - **task_id**: Task ID returned from upload endpoint
    """
    
    if task_id not in tasks_db: 
        raise HTTPException(status_code = 404 , detail = "Task Not Found.")

    task_data = tasks_db[task_id]

    return TaskStatusResponse(
        task_id=task_data["task_id"],
        status=task_data["status"],
        progress=task_data["progress"],
        message=task_data.get("message"),
        result=task_data.get("result"),
        error=task_data.get("error"),
        created_at=task_data["created_at"],
        completed_at=task_data.get("completed_at")
    )


@router.get("/result/{task_id}" , response_model = AnalysisResult)
async def get_analysis_result(task_id : str):
    """
    Get complete analysis result for a completed task
    
    - **task_id**: Task ID returned from upload endpoint
    """
    
    if task_id not in tasks_db: 
        raise HTTPException(status_code = 404 , detail = "Task Not Found.")

    task_data = tasks_db[task_id]

    if task_data["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code = 400 , detail = f"Task Not Completed. Current Status : {task_data['status']}")

    result = task_data["result"]
    stats = result["statistics"]

    return AnalysisResult(
        task_id=task_data["task_id"],
        video_url= result["video_url"],
        statistics= Statistics(**stats),
        heatmap_urls=[],
        formation_data=None
    )

@router.get("/download/{task_id}/{filename}")
async def download_file(task_id : str , filename : str):
    """
    Download processed video or analysis files
    
    - **task_id**: Task ID
    - **filename**: Name of file to download
    """

    file_path = settings.OUTPUT_DIR / task_id / filename

    if not file_path.exists():
        raise HTTPException(status_code = 404 , detail = "File Not Found.")

    return FileResponse(
        path = str(file_path) , 
        filename = filename,
        media_type = "application/octet-stream"
    )
    

@router.delete("/task/{task_id}")
async def delete_task(task_id : str):
    """
    Delete a task and its associated files
    
    - **task_id**: Task ID to delete
    """

    if task_id not in tasks_db:
        raise HTTPException(status_code = 404 , detail = "Task Not Found.")

    upload_dir = settings.UPLOAD_DIR / task_id
    output_dir = settings.OUTPUT_DIR / task_id

    shutil.rmtree(upload_dir , ignore_errors = True)
    shutil.rmtree(output_dir , ignore_errors = True)

    del tasks_db[task_id]

    return {"message": "Task deleted successfully", "task_id": task_id}