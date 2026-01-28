from pydantic import BaseModel , Field
from typing import Optional , List , Dict
from datetime import datetime
from enum import Enum


class TaskStatus(str,Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoUploadResponse(BaseModel):
    task_id : str
    status : TaskStatus
    message : str
    created_at : datetime

class TaskStatusResponse(BaseModel):
    task_id : str
    status : TaskStatus
    progress : int = Field(ge = 0 , le = 100)
    message : Optional[str] = None
    result : Optional[Dict] = None
    error : Optional[str] = None
    created_at : datetime
    completed_at : Optional[datetime] = None

class Statistics(BaseModel):
    total_frames : int
    total_players_tracked : int
    team_1_possession_percent : float
    team_2_possession_percent : float
    processing_time_seconds : float
    average_player_speed_kmh : Optional[float] = None
    total_distance_covered_m : Optional[float] = None
    
class AnalysisResult(BaseModel):
    task_id : str
    video_url : str
    thumbnail_url : Optional[str] = None
    statistics : Statistics
    heatmap_urls : List[str]  = []
    formation_data : Optional[Dict] = None
