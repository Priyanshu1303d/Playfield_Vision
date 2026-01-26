from src.Playfield_Vision.utils.video_utils import read_videos , save_video
from src.Playfield_Vision.trackers.tracker import Tracker
import cv2
from src.Playfield_Vision.team_assigner.team_assigner import TeamAssigner
from src.Playfield_Vision.player_ball_assigner.player_ball_assigner import PlayerBallAssigner
import numpy as np
from src.Playfield_Vision.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from src.Playfield_Vision.view_transformer.view_transformer import ViewTransformer 
from src.Playfield_Vision.speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
   "Read Video"

   video_frames = read_videos('Input_Videos\\B1606b0e6_1_17.mp4')

   # Initialize the tracker
   tracker = Tracker("models\\best_yolov5.pt")
   
   tracks = tracker.get_object_tracks(video_frames  , False , "stubs\\track_stubs.pkl")

   #get Object positions
   tracker.add_positions_to_tracks(tracks)
   

   # save cropped image of the player
   # for track_id , player in tracks["player"][0].items():
   #    frame = video_frames[0]
   #    bbox = player["bbox"]
   #    x1 , y1 , x2 , y2 = bbox
   #    cropped_image = frame[int(y1):int(y2) , int(x1):int(x2)]
   #    cv2.imwrite(f"output_videos\\save_video_demo\\cropped_image_{track_id}.jpg" , cropped_image)
   #    break


   # Camera movemtent estimator
   camera_movement_estimator = CameraMovementEstimator(video_frames)
   camera_movement_per_frame = camera_movement_estimator.get_camera_movement_estimator(
      video_frames,
      read_from_stub = False,
      stub_path = "stubs\\camera_movement_stubs.pkl"
   )
   camera_movement_estimator.add_adjust_positions_to_tracks(tracks , camera_movement_per_frame)

   #View transformer
   view_transformer = ViewTransformer()
   view_transformer.add_transformed_positions_to_tracks(tracks)

   #Interpolate ball positions
   tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])


   #Speed and distance estimator
   speed_and_distance_estimator = SpeedAndDistance_Estimator()
   speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
   


   #Assign team color
   team_assigner = TeamAssigner()
   team_assigner.assign_team_color(video_frames[0] , tracks['player'][0])

   for frame_num, player_tracks in enumerate(tracks["player"]):
      for player_id , track in player_tracks.items():
         team = team_assigner.get_player_team(video_frames[frame_num] , track["bbox"] , player_id)

         track["team"] = team
         track["team_color"] = team_assigner.team_colors[team]

   
   # Assign ball Acquisition
   player_assigner = PlayerBallAssigner()
   team_ball_control = []
   for frame_num ,  player_tracks in enumerate(tracks['player']):
      ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])
      assigned_player = player_assigner.assign_ball_to_player(player_tracks , ball_bbox)

      if assigned_player != -1:
         tracks['player'][frame_num][assigned_player]['has_ball'] = True
         team_ball_control.append(tracks['player'][frame_num][assigned_player]['team']) 
      else:
         team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

   team_ball_control = np.array(team_ball_control)

   ##Draw output 
   #Draw object tracks
   output_video_frames = tracker.draw_annotations(video_frames , tracks , team_ball_control)


   #Draw camera movement
   output_video_frames = camera_movement_estimator.draw_camera_movement(
      output_video_frames,
      camera_movement_per_frame,
   )

   #Draw speed and distance
   output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames , tracks)

   #Save Video
   save_video(output_video_frames , "output_videos\\save_video_demo\\output.avi")

if __name__ == '__main__' : 
    main()