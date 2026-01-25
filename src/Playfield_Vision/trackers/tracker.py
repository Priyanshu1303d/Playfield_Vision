from ultralytics import YOLO
import pandas as pd
import supervision as sv
import pickle
import os
from src.Playfield_Vision.utils.bbox_utils import get_center_of_bbox , get_bbox_width
import cv2
import numpy as np


class Tracker :
    def __init__(self , model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get("bbox",[]) for x in ball_positions]
        
        # If no ball detections exist, return empty list for each frame
        if not ball_positions or all(len(x) == 0 for x in ball_positions):
            return [{} for _ in range(len(ball_positions))]
        
        df_ball_positions = pd.DataFrame(ball_positions, columns = ["x1" , "y1" , "x2" , "y2"])

        #interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1 : {"bbox" : x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self , frames):
        batch_size = 20
        detections = []

        for i in range(0 , len(frames) , batch_size):
            detection_batch = self.model.predict(frames[i : i + batch_size] , conf= 0.1)
            detections += detection_batch

        return detections
    

    def get_object_tracks(self , frames, read_from_stub =False, stub_path = None)  :

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "player" : [],
            # eg "player" : [
            #   {0 : {"bbox" : [x1 , y1 , x2 , y2]} , 1 : {"bbox" : [x1 , y1 , x2 , y2]}} #Tracks All the player in frame 1  
            #   {10 : {"bbox" : [x1 , y1 , x2 , y2]} , 12 : {"bbox" : [x1 , y1 , x2 , y2]}} #Tracks All the player in frame 2  
            # ],
            "ball" : [],
            "referee" : [],
        }

        for frame_num , detection in enumerate(detections):
            class_names = detection.names
            class_names_inversion = {v:k for k , v in class_names.items()}

            print(class_names)

            #Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert goalkeeper to player
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = class_names_inversion['player']


            #Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            print(detection_with_tracks)

            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["referee"].append({})


            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == class_names_inversion['player']:
                    tracks["player"][frame_num][track_id] = {"bbox" : bbox}

                if cls_id == class_names_inversion['referee']:
                    tracks["referee"][frame_num][track_id] = {"bbox" : bbox}


            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == class_names_inversion.get('ball'):
                    tracks["ball"][frame_num][1] = {"bbox" : bbox}
                    print(f"✓ Ball detected in frame {frame_num}: {bbox}")
            
            # Debug: Show what was detected in first frame
            if frame_num == 0:
                detected_classes = [class_names[cls_id] for cls_id in detection_supervision.class_id]
                print(f"\n=== Frame 0 Debug ===")
                print(f"Available classes: {class_names}")
                print(f"Detected classes: {set(detected_classes)}")
                print(f"Class counts: {[(cls, detected_classes.count(cls)) for cls in set(detected_classes)]}")
                print(f"'ball' in class_names_inversion: {'ball' in class_names_inversion}")
                print(f"==================\n")

        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_epllipse(self , frame , bbox , color , track_id = None):
        y2 = int(bbox[3])
        
        x_center , _ = get_center_of_bbox(bbox)
        bbox_width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center = (x_center , y2),
            axes = (int(bbox_width) , int(0.35*bbox_width)),
            angle = 0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType= cv2.LINE_4
        )

        #Drawing rectangle
        rectangle_width = 40
        rectangle_height = 20

        x1_react = x_center - rectangle_width // 2
        x2_react = x_center + rectangle_width // 2
        y1_react = (y2 - rectangle_height // 2) + 15
        y2_react = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (x1_react , y1_react),
                (x2_react , y2_react),
                color,
                cv2.FILLED,
            )

            x1_text = x1_react + 12
            if track_id > 99 :
                x1_text -= 10

            cv2.putText(
                frame,
                str(track_id),
                (int(x1_text) , int(y1_react + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2,
                cv2.LINE_4
            )

        return frame

    def draw_triangle(self, frame , bbox, color):
        y = int(bbox[1])
        x_center , _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            (x_center , y),
            (x_center - 10 , y - 20),
            (x_center + 10 , y - 20)
        ])


        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)        
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) 

        return frame

    def draw_team_ball_control(self , frame ,frame_num , team_ball_control):
        #Draw a semi-transparent rectangle
        overlay = frame.copy()

        cv2.rectangle(overlay , (1350, 850), (1900, 970), (255,255,255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay , alpha , frame , 1 - alpha , 0 , frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        #Get the Number of times each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]


        team_1_percentage = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2_percentage = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame , f"Team 1 Ball Control: {team_1_percentage * 100:.2f}%" , (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame , f"Team 2 Ball Control: {team_2_percentage * 100:.2f}%" , (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        
        return frame


    def draw_annotations(self , video_frame , tracks , team_ball_control) :
        output_video_frames = []

        for frame_num , frame in enumerate(video_frame):
            frame = frame.copy()

            player_dict = tracks["player"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]

            for track_id , player in player_dict.items():
                team_color = player.get("team_color", (0,0,255))
                frame = self.draw_epllipse(frame, player["bbox"], team_color , track_id)

                if player.get("has_ball" , False):
                    frame = self.draw_triangle(frame, player["bbox"] , (0,0,255))

            
            for _ , refree in referee_dict.items():
                frame = self.draw_epllipse(frame, refree["bbox"], (0,255,255))

            
            for _ , ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            # Draw Team ball control
            frame = self.draw_team_ball_control(frame ,frame_num , team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames