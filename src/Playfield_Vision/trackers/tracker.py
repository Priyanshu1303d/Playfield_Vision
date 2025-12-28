from ultralytics import YOLO
import supervision as sv



class Tracker :
    def __init__(self , model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self , frames):
        batch_size = 20
        detections = []

        for i in range(0 , len(frames) , batch_size):
            detection_batch = self.model.predict(frames[i : i + batch_size] , conf= 0.1)
            detections += detection_batch

            break

        return detections
    

    def get_object_tracks(self , frames):

        detections = self.detect_frames(frames)

        for frame_num , detection in enumerate(detections):
            class_names = detection.names
            class_names_inversion = {v:k for k , v in class_names.items()}

            #Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)




