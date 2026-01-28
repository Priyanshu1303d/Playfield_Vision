from pathlib import Path
from pydantic_settings import BaseSettings 


class Settings(BaseSettings):
    """Application Settings"""
    
    APP_NAME : str = "Playfield_Vision"
    DEBUG : bool = True
    VERSION : str = "1.0.0"
 
    BASE_DIR: Path =  Path(__file__).parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    MODEL_PATH: Path = BASE_DIR / "models" / "best_yolov5.pt"

    MAX_FILE_SIZE : int = 500 * 1024 * 1024
    ALLOWED_EXTENSIONS : set = {".mp4", ".avi", ".mov", ".mkv"}

    CONFIDENCE_THRESHOLD : float = 0.5
    FRAME_RATE : int = 24
    BATCH_SIZE : int = 20

    USE_STUB_CACHE  : bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

settings.UPLOAD_DIR.mkdir(parents = True , exist_ok = True)
settings.OUTPUT_DIR.mkdir(parents = True , exist_ok = True)