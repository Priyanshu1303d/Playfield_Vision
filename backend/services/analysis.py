import os
import time
import traceback
from pathlib import Path
from typing import Dict , Any
import numpy as np

from src.Playfield_Vision.utils.video_utils import read_videos, save_video
from src.Playfield_Vision.trackers.tracker import Tracker
from src.Playfield_Vision.team_assigner.team_assigner import TeamAssigner
from src.Playfield_Vision.player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from src.Playfield_Vision.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from src.Playfield_Vision.view_transformer.view_transformer import ViewTransformer
from src.Playfield_Vision.speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator

from backend.config import settings


class VideoAnalysisService:
    """
    Service to process uploaded videos through the complete analysis pipeline.
    """
    
    def __init__(self):
        self.tracker = Tracker(str(settings.MODEL_PATH))

    def process_video(self, video_path : str , output_dir : str , task_id : str) -> Dict[str, Any]:
        """
        Process uploaded video through complete pipeline
        
        Args:
            video_path: Path to uploaded video file
            output_dir: Directory to save outputs
            task_id: Unique task identifier
            
        Returns:
            Dictionary containing analysis results and file paths
        """

        start_time = time.time()

        try:
            os.makedirs(output_dir , exist_ok = True)
            print(f"[{task_id}] Reading video frames...")
            video_frames = read_videos(video_path)
            total_frames = len(video_frames)

            if total_frames == 0:
                return {
                    "success": False,
                    "error": "No frames could be extracted from video"
                }
            
            print(f"[{task_id}] Tracking objects in {total_frames} frames...")
            stub_path = os.path.join(output_dir, "track_stubs.pkl")
            tracks = self.tracker.get_object_tracks(
                video_frames, 
                read_from_stub=settings.USE_STUB_CACHE, 
                stub_path=stub_path
            )

            print(f"[{task_id}] Adding positions to tracks...")
            self.tracker.add_positions_to_tracks(tracks)
            
            print(f"[{task_id}] Estimating camera movement...")
            camera_estimator = CameraMovementEstimator(video_frames)
            camera_movement = camera_estimator.get_camera_movement_estimator(
                video_frames,
                read_from_stub=settings.USE_STUB_CACHE,
                stub_path=os.path.join(output_dir, "camera_stubs.pkl")
            )
            camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
            
            print(f"[{task_id}] Transforming view perspective...")
            view_transformer = ViewTransformer()
            view_transformer.add_transformed_positions_to_tracks(tracks)
            
            print(f"[{task_id}] Interpolating ball positions...")
            tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])
            
            print(f"[{task_id}] Calculating speed and distance...")
            speed_estimator = SpeedAndDistance_Estimator()
            speed_estimator.add_speed_and_distance_to_tracks(tracks)
            
            print(f"[{task_id}] Assigning team colors...")
            team_assigner = TeamAssigner()
            team_assigner.assign_team_color(video_frames[0], tracks['player'][0])
            
            for frame_num, player_tracks in enumerate(tracks["player"]):
                for player_id, track in player_tracks.items():
                    team = team_assigner.get_player_team(
                        video_frames[frame_num], 
                        track["bbox"], 
                        player_id
                    )
                    track["team"] = team
                    track["team_color"] = team_assigner.team_colors[team]
            
            print(f"[{task_id}] Analyzing ball possession...")
            player_assigner = PlayerBallAssigner()
            team_ball_control = []
            
            for frame_num, player_tracks in enumerate(tracks['player']):
                ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])
                assigned_player = player_assigner.assign_ball_to_player(
                    player_tracks, 
                    ball_bbox
                )
                
                if assigned_player != -1:
                    tracks['player'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(
                        tracks['player'][frame_num][assigned_player]['team']
                    )
                else:
                    team_ball_control.append(
                        team_ball_control[-1] if team_ball_control else 1
                    )
            
            team_ball_control = np.array(team_ball_control)
            
            print(f"[{task_id}] Drawing annotations...")
            output_frames = self.tracker.draw_annotations(
                video_frames, 
                tracks, 
                team_ball_control
            )
            output_frames = camera_estimator.draw_camera_movement(
                output_frames, 
                camera_movement
            )
            output_frames = speed_estimator.draw_speed_and_distance(
                output_frames, 
                tracks
            )
            
            print(f"[{task_id}] Saving output video...")
            output_video_path = os.path.join(output_dir, "analyzed_video.avi")
            save_video(output_frames, output_video_path)
            
            print(f"[{task_id}] Calculating statistics...")
            stats = self._calculate_statistics(
                tracks, 
                team_ball_control, 
                total_frames, 
                start_time
            )


            return {
                "success": True,
                "output_video": output_video_path,
                "statistics": stats,
                "tracks": tracks,
                "team_ball_control": team_ball_control.tolist()
            }

        except Exception as e:
            print(f"[{task_id}] ERROR: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    
    def _calculate_statistics(
        self, 
        tracks : Dict, 
        team_ball_control : np.ndarray, 
        total_frames : int, 
        start_time : float
    ) -> Dict[str , Any]:
        """
        Calculate statistics for the video analysis.
        
        Args:
            tracks: Dictionary of tracks
            team_ball_control: Array of team ball control
            total_frames: Total number of frames
            start_time: Start time of the analysis
        
        Returns:
            Dictionary containing statistics
        """


        team_1_frames = np.sum(team_ball_control == 1)
        team_2_frames = np.sum(team_ball_control == 2)

        total_frames_count = team_1_frames + team_2_frames

        
        if total_frames_count > 0:
            team_1_possession_percent = (team_1_frames / total_frames_count) * 100
            team_2_possession_percent = (team_2_frames / total_frames_count) * 100
        else:
            team_1_possession_percent = 0
            team_2_possession_percent = 0

        all_player_ids = set()
        total_distance = 0
        speeds = []

        for frame in tracks['player']:
            for player_id, track_info in frame.items():
                all_player_ids.add(player_id)
                
                if 'speed' in track_info:
                    speeds.append(track_info['speed'])
                
                if 'distance' in track_info:
                    total_distance = max(total_distance, track_info['distance'])
        
        avg_speed = np.mean(speeds) if speeds else None
        
        return {
            "total_frames": total_frames,
            "total_players_tracked": len(all_player_ids),
            "team_1_possession_percent": round(team_1_possession_percent, 2),
            "team_2_possession_percent": round(team_2_possession_percent, 2),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "average_player_speed_kmh": round(avg_speed, 2) if avg_speed else None,
            "total_distance_covered_m": round(total_distance, 2)
        }
        