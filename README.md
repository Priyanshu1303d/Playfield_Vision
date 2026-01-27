# Playfield Vision: End-to-End Football Vision Analytics 

An AI-powered football/soccer analysis system that uses computer vision and machine learning to track players, detect the ball, assign team colors, and analyze ball possession in real-time from video footage.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Detection-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Features

- **🎥 Player Tracking**: Real-time multi-object tracking using YOLOv5 and ByteTrack
- **👕 Team Color Assignment**: Automatic team identification using K-Means color clustering
- **⚽ Ball Detection & Interpolation**: Smooth ball tracking with position interpolation for missing frames
- **🎯 Ball Possession Analysis**: Identifies which player has the ball at each frame
- **📊 Team Ball Control Statistics**: Real-time percentage calculation of ball possession per team
- **🎨 Visual Annotations**: Color-coded ellipses for players, triangles for ball, and on-screen statistics

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV, Ultralytics YOLOv5
- **Object Tracking**: Supervision (ByteTrack)
- **Machine Learning**: scikit-learn (K-Means clustering)
- **Data Processing**: NumPy, Pandas
- **Deep Learning Framework**: PyTorch (via Ultralytics)

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Priyanshu1303d/Playfield_Vision.git
   cd Playfield_Vision
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download YOLOv5 model**
   - Place your trained `best_yolov5.pt` model in the `models/` directory
   - Or train your own model using the training pipeline

## 🚀 Usage

### Basic Usage

1. **Prepare your input video**
   - Place your football match video in the `Input_Videos/` directory

2. **Run the analysis**
   ```bash
   python main.py
   ```

3. **Output**
   - Annotated video will be saved in `output_videos/save_video_demo/output.avi`
   - Tracking data will be cached in `stubs/track_stubs.pkl` for faster reprocessing

### Configuration Options

Edit `main.py` to customize:

- **Input video path**: Change line 11
  ```python
  video_frames = read_videos('Input_Videos\\your_video.mp4')
  ```

- **Model path**: Change line 14
  ```python
  tracker = Tracker("models\\your_best_model.pt")
  ```

- **Cache behavior**: Set `read_from_stub` to `False` for fresh detection (line 16)
  ```python
  tracks = tracker.get_object_tracks(video_frames, False, "stubs\\track_stubs.pkl")
  ```

- **Output path**: Change line 63
  ```python
  save_video(output_video_frames, "output_videos\\your_output.avi")
  ```

## 📁 Project Structure
```
Playfield_Vision/
│
├── backend/                         # FastAPI backend (API + orchestration)
│   ├── __init__.py
│   ├── main.py                      # FastAPI entry point
│   ├── config.py                    # Env vars, paths, constants
│   ├── schemas.py                   # Pydantic request/response models
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   └── video.py                 # Upload video → process → return results
│   │
│   └── services/
│       ├── __init__.py
│       ├── pipeline.py              # Calls core Playfield_Vision logic
│       ├── analysis.py              # Metrics, stats, possession, etc.
│       ├── heatmap.py               # Heatmap generation
│       └── report.py                # PDF / analytics report generation
│
├── frontend/                        # Streamlit frontend
│   ├── streamlit_app.py             # Main Streamlit app
│   │
│   ├── components/
│   │   ├── uploader.py              # Video upload UI
│   │   ├── dashboard.py             # Metrics, charts, KPIs
│   │   └── visualizations.py        # Heatmaps, trajectories
│   │
│   └── utils.py                     # API calls, helpers
│
├── src/
│   └── Playfield_Vision/            # CORE ML / CV LOGIC
│       ├── trackers/                # Object detection & tracking
│       │   └── tracker.py
│       │
│       ├── team_assigner/           # Team color clustering
│       │   └── team_assigner.py
│       │
│       ├── player_ball_assigner/    # Ball possession logic
│       │   └── player_ball_assigner.py
│       │
│       ├── utils/                   # Helper functions
│       │   ├── video_utils.py
│       │   └── bbox_utils.py
│       │
│       ├── camera_movement_estimator/        # (Future)
│       ├── speed_and_distance_estimator/     # (Future)
│       ├── view_transformer/                 # (Future)
│       └── __init__.py
│
├── models/                          # YOLO / trained weights
│
├── data/
│   ├── input_videos/                # Uploaded raw videos
│   ├── output_videos/               # Processed videos
│   └── stubs/                       # Cached tracking data
│
├── research/                        # Jupyter notebooks / experiments
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
│
├── requirements.txt
├── README.md
└── .env
```

## 🔍 How It Works

### 1. Object Detection & Tracking
- Uses **YOLOv5** trained on custom football dataset to detect players, ball, and referees
- **ByteTrack** algorithm maintains consistent player IDs across frames
- Converts goalkeepers to player class for unified tracking

### 2. Team Assignment
- Extracts player jersey colors from bounding box regions
- Uses **K-Means clustering** to identify two dominant team colors
- Assigns each player to a team based on color similarity
- Caches assignments for performance

### 3. Ball Detection & Interpolation
- Detects ball in each frame using YOLO
- Uses **pandas interpolation** to fill missing ball positions
- Handles occlusions and temporary ball disappearances

### 4. Ball Possession Analysis
- Calculates Euclidean distance from ball to each player
- Assigns ball to nearest player within threshold distance
- Tracks possession changes frame-by-frame

### 5. Visual Annotations
- **Players**: Color-coded ellipses with track IDs (team colors)
- **Ball**: Green triangle
- **Player with ball**: Red triangle overlay
- **Referees**: Yellow ellipse
- **Statistics overlay**: Semi-transparent box with ball control percentages

## 📊 Sample Output

The system generates annotated videos with:
- ✅ Player tracking with unique IDs
- ✅ Team differentiation via color
- ✅ Ball position tracking
- ✅ Real-time ball possession stats
- ✅ Team ball control percentages

## 🧪 Testing & Development

### Running with stub cache
```python
# Enable for faster testing after first run
tracks = tracker.get_object_tracks(video_frames, True, "stubs\\track_stubs.pkl")
```

### Debug mode
Uncomment debug print statements in `tracker.py` to see:
- Detected classes per frame
- Ball detection confirmations
- Tracking information

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv5
- **Roboflow** for dataset management
- **Supervision** library for tracking algorithms
- Football analytics community for inspiration

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made by Priyanshu Kumar Singh**
