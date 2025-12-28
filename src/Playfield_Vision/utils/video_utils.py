import cv2


def read_videos(video_filepath : str) ->  list:
    '''
        Converts the video into frames
        Args:
            video_filepath : String
        Returns: 
            list of frames extracted from the video
    '''

    cap = cv2.VideoCapture(video_filepath)

    frames = []

    while True:
        ret , frame = cap.read()
        if not ret :
            break

        frames.append(frame) 

    return frames


def save_video(output_video_frames : list , output_video_path : str) :
    '''
        Converts the list of frames to the video and saves it to output video path location
        Args:
            output_video_frames : list
            output_video_path : str
    '''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_video_path , fourcc , 24 , (output_video_frames[0].shape[1] , output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print(f"Video saved successfully at: --->  {output_video_path}")
