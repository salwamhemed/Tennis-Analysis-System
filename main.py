from utils import (read_video, save_video,measure_distance,convert_pixel_distance_to_meters, draw_player_stats)
from tracking import player_tracker 
from tracking import ball_tracker
from court_line_detection import CourtLineDetector
import cv2
from mini_court import MiniCourt
import constants
from copy import deepcopy
import pandas as pd

def main():
    #Reading the video
    path = r"C:\Users\salwa\OneDrive\Desktop\tennis project\input\input_video.mp4"
    frames = read_video(path)
    

    #Detecting the players
    PlayerTracker = player_tracker(model_path='yolov8x') 
    player_detection = PlayerTracker.detect_frames(frames, read_from_stub=True, stub_path=r"C:\Users\salwa\OneDrive\Desktop\tennis project\tracking_stubs\player_detections.pkl")
   

    #Detect the ball
    BallTracker = ball_tracker(model_path=r"C:\Users\salwa\OneDrive\Desktop\tennis project\models\last.pt")
    ball_detection = BallTracker.detect_frames(frames, read_from_stub=True, stub_path=r"C:\Users\salwa\OneDrive\Desktop\tennis project\tracking_stubs\ball_detection.pkl")
    ball_detection = BallTracker.interpolate_ball_position(ball_detection)


    #get the ball shot frames
    ball_shot_frames = BallTracker.get_ball_shot_frames(ball_detection)
    print(ball_shot_frames)


    #Detect the courtLine
    court_Line_Detector= CourtLineDetector(model_path=r"C:\Users\salwa\OneDrive\Desktop\tennis project\models\keypoints_model.pth")
    court_keypoints = court_Line_Detector.predict(frames[0])


    #choose the players
    player_detection = PlayerTracker.choose_and_filter_players(court_keypoints , player_detection)
    

    #Drawing player bounding boxes 
    output_video_frames = PlayerTracker.draw_bboxes(frames, player_detection)
    output_video_frames = BallTracker.draw_bboxes(output_video_frames, ball_detection)
  
    #Drawing court key points
    output_video_frames = court_Line_Detector.draw_keypoints_on_video(output_video_frames,court_keypoints)


    # Detect Mini court
    mini_court = MiniCourt(frames[0])
  
    #convert positions to mini court positions 
    players_mini_court_detection , ball_mini_court_detection = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detection, ball_detection, court_keypoints)
  
  
    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,players_mini_court_detection)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detection, color=(0,0,0))  
 
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_latest_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_latest_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_latest_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_latest_player_speed':0,
    } ]

    for ball_shot_index in range(len(ball_shot_frames)-1) : 

        start_frame = ball_shot_frames[ball_shot_index]
        end_frame = ball_shot_frames[ball_shot_index+1]
        ball_shot_time_in_seconds = (start_frame - end_frame)/24 #because each 1s we have 24 pixels

        #Distance covered by the ball
        distance_covered_by_the_ball_in_pixels = measure_distance(ball_mini_court_detection[start_frame][1], ball_mini_court_detection[end_frame][1]) #y 

        distance_covered_by_the_ball_in_meters = convert_pixel_distance_to_meters(distance_covered_by_the_ball_in_pixels, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court() )

        #speed of the ball shot in km/h
        speed_of_the_ball_shot = (distance_covered_by_the_ball_in_meters / ball_shot_time_in_seconds) *3.6 

        #Player who shot the ball
        player_positions = players_mini_court_detection[start_frame]
        player_who_shot_the_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detection[start_frame][1]))
        #oppenent player 

        oppenent_player_id = 1 if player_who_shot_the_ball==2 else 2 
        
        distance_covered_by_oppenent_in_pixels = measure_distance(players_mini_court_detection[start_frame][oppenent_player_id], players_mini_court_detection[end_frame][oppenent_player_id])
        distance_covered_by_oppenent_in_meters = convert_pixel_distance_to_meters(distance_covered_by_oppenent_in_pixels,constants.DOUBLE_LINE_WIDTH , mini_court.get_width_of_mini_court())

        #speed of the oppenent player 

        oppenent_player_speed = ( distance_covered_by_oppenent_in_meters / ball_shot_time_in_seconds) * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_who_shot_the_ball}_number_of_shots"] += 1 
        current_player_stats[f"player_{player_who_shot_the_ball}_total_shot_speed"] += speed_of_the_ball_shot
        current_player_stats[f"player_{player_who_shot_the_ball}_latest_shot_speed"] = speed_of_the_ball_shot

        current_player_stats[f"player_{oppenent_player_id}_total_player_speed"] += oppenent_player_speed
        current_player_stats[f"player_{oppenent_player_id}_latest_player_speed"] =  oppenent_player_speed

        player_stats_data.append(current_player_stats)


    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']


    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    #Drawing the frame numbers in the video
    for i,frame in enumerate(output_video_frames):
        cv2.putText(frame,f" frame : {i}" ,(10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255),2)
    save_video(output_video_frames, r"C:\Users\salwa\OneDrive\Desktop\tennis project\saved_videos\final_video.avi")

if __name__ == "__main__":
    main()
