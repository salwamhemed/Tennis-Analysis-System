from utils import (read_video, save_video)
from tracking import player_tracker 
from tracking import ball_tracker
from court_line_detection import CourtLineDetector
def main():
    #Reading the video
    path = r"C:\Users\salwa\OneDrive\Desktop\tennis project\input\input_video.mp4"
    frames1 = read_video(path)
    

    #Detecting the players
    PlayerTracker = player_tracker(model_path='yolov8x') 
    player_detection = PlayerTracker.detect_frames(frames1, read_from_stub=True, stub_path=r"C:\Users\salwa\OneDrive\Desktop\tennis project\tracking_stubs\player_detections.pkl")
    #Detect the ball

    BallTracker = ball_tracker(model_path=r"C:\Users\salwa\OneDrive\Desktop\tennis project\models\last.pt")
    ball_detection = BallTracker.detect_frames(frames1, read_from_stub=True, stub_path=r"C:\Users\salwa\OneDrive\Desktop\tennis project\tracking_stubs\ball_detection.pkl")
    
    #Detect the courtLine
    court_Line_Detector= CourtLineDetector(r"C:\Users\salwa\OneDrive\Desktop\tennis project\models\model_tennis_court_det.pt")
    court_keypoints = court_Line_Detector.predict(frames1[0])

    
    #Drawing player bounding boxes 
    output_video_frames = PlayerTracker.draw_bboxes(frames1, player_detection)
    output_video_frames = BallTracker.draw_bboxes(output_video_frames, ball_detection)
    #Drawing cour key points
    output_video_frames = court_Line_Detector.draw_keypoints_on_video(output_video_frames,court_keypoints)


    save_video(output_video_frames, r"C:\Users\salwa\OneDrive\Desktop\tennis project\saved_videos\saved_video7.avi")

if __name__ == "__main__":
    main()