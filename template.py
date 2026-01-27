import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] : %(message)s")

project_name = "Playfield_Vision"

list_of_files = [
    # Root level config files
    ".env",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "setup.py",
    
    # Docker files
    "Dockerfile.backend",
    "Dockerfile.frontend",
    "docker-compose.yml",
    ".dockerignore",
    
    
    "backend/__init__.py",
    "backend/main.py",
    "backend/config.py",
    "backend/schemas.py",
    
    # Backend routes
    "backend/routes/__init__.py",
    "backend/routes/video.py",
    
    # Backend services
    "backend/services/__init__.py",
    "backend/services/analysis.py",
    "backend/services/heatmap.py", 
    "backend/services/formation.py",
    
    # Backend utilities
    "backend/utils/__init__.py",
    "backend/utils/logger.py",
    
    # Backend tests
    "backend/tests/__init__.py",
    "backend/tests/test_api.py",
    "backend/tests/performance_test.py",
    

    "frontend/streamlit_app.py",
    "frontend/utils.py",
    
    # Frontend components
    "frontend/components/__init__.py",
    "frontend/components/dashboard.py",
    "frontend/components/visualizations.py",
    

    f"src/{project_name}/__init__.py",
    

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/video_utils.py",
    f"src/{project_name}/utils/bbox_utils.py",
    

    f"src/{project_name}/trackers/__init__.py",
    f"src/{project_name}/trackers/tracker.py",
    

    f"src/{project_name}/team_assigner/__init__.py",
    f"src/{project_name}/team_assigner/team_assigner.py",
    

    f"src/{project_name}/player_ball_assigner/__init__.py",
    f"src/{project_name}/player_ball_assigner/player_ball_assigner.py",
    
    f"src/{project_name}/camera_movement_estimator/__init__.py",
    f"src/{project_name}/camera_movement_estimator/camera_movement_estimator.py",
    
    f"src/{project_name}/view_transformer/__init__.py",
    f"src/{project_name}/view_transformer/view_transformer.py",
    f"src/{project_name}/speed_and_distance_estimator/__init__.py",
    f"src/{project_name}/speed_and_distance_estimator/speed_and_distance_estimator.py",

    "main.py",
    "app.py",

    "research/01_yolo_inference.ipynb",
    "research/02_color_segmentation.ipynb",
    

    "Input_Videos/.gitkeep",
    "output_videos/.gitkeep",
    "models/.gitkeep",
    "stubs/.gitkeep",
    "uploads/.gitkeep",
    "outputs/.gitkeep",
    "logs/.gitkeep",
]

def create_project_structure():
    """Create complete project structure for Playfield Vision"""
    
    created_folders = set()
    created_files = []
    existing_files = []
    
    for file_path_str in list_of_files:
        file_path = Path(file_path_str)
        folder = file_path.parent

        if folder != Path(".") and str(folder) not in created_folders:
            folder.mkdir(parents=True, exist_ok=True)
            created_folders.add(str(folder))
            logging.info(f"📁 Folder created: {folder}")
        
        # Create file if it doesn't exist or is empty
        if not file_path.exists() or file_path.stat().st_size == 0:
            with open(file_path, "w") as f:
                # Add helpful comment for .gitkeep files
                if file_path.name == ".gitkeep":
                    f.write("# This file keeps the directory in git\n")
            
            created_files.append(str(file_path))
            logging.info(f"✅ Created: {file_path}")
        else:
            existing_files.append(str(file_path))
            logging.info(f"⏭️  Already exists: {file_path}")
    
    print("\n" + "="*60)
    print("📊 PROJECT STRUCTURE SUMMARY")
    print("="*60)
    print(f"📁 Folders created: {len(created_folders)}")
    print(f"✅ Files created: {len(created_files)}")
    print(f"⏭️  Files skipped (already exist): {len(existing_files)}")
    print("="*60)
    

if __name__ == '__main__':
    create_project_structure()