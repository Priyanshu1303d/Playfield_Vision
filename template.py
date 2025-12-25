import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] : %(message)s")

project_name = "Playfield_Vision"

list_of_files = [
    ".env",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "setup.py",
    "dvc.yaml",
    
    f"src/{project_name}/__init__.py",
    
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/video_utils.py", # Reading/Saving video
    f"src/{project_name}/utils/bbox_utils.py",  # Drawing bounding boxes
    
    f"src/{project_name}/trackers/__init__.py",
    f"src/{project_name}/trackers/tracker.py", # YOLO + ByteTrack logic
    
    f"src/{project_name}/team_assigner/__init__.py",
    f"src/{project_name}/team_assigner/team_assigner.py", # K-Means color clustering
    
    f"src/{project_name}/camera_movement_estimator/__init__.py",
    f"src/{project_name}/camera_movement_estimator/camera_movement_estimator.py", # Optical Flow
    
    f"src/{project_name}/view_transformer/__init__.py",
    f"src/{project_name}/view_transformer/view_transformer.py", # Perspective Transformation
    
    f"src/{project_name}/speed_and_distance_estimator/__init__.py",
    f"src/{project_name}/speed_and_distance_estimator/speed_and_distance_estimator.py",
    
    f"src/main.py", 
    f"src/app.py"
    
    "app/routes.py",
    "app/schemas.py",
    
    "research/01_yolo_inference.ipynb",
    "research/02_color_segmentation.ipynb",
    
    "input_videos/test_video.mp4",
    "output_videos/demo_video.py",
    "models/demo.py",           # To store .pt files
    "stubs/demo.py"             # To store cache/pkl files for speed
]

for i in list_of_files:
    file_path = Path(i)
    folder, filename = os.path.split(file_path)

    if folder != "":
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Folder created: {folder}")

    if (not file_path.exists()) or (file_path.stat().st_size == 0):
        with open(file_path, "w") as f:
            pass
        logging.info(f"Created empty file: {file_path}")

    else:
        logging.info(f"File already exists: {file_path}")