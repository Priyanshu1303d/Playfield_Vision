from src.Playfield_Vision.utils.video_utils import read_videos , save_video
from src.Playfield_Vision.trackers.tracker import Tracker


def main():
   "Read Video"

   video_frames = read_videos('Input_Videos\\B1606b0e6_1_17.mp4')

   # Initialize the tracker
   tracker = Tracker("models\\best_yolov5.pt")
   
   tracks = tracker.get_object_tracks(video_frames)


   #Save Video
   save_video(video_frames , "output_videos\\save_video_demo\\output.avi")

if __name__ == '__main__' : 
    main()