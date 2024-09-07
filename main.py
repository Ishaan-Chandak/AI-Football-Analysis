from utils import read_video, save_video
from trackers import Tracker
from player_team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_distance_estimator import SpeedDistanceEstimator
import cv2
import numpy as np

def main(): 
    # read video
    video_frames = read_video('inputs/08fd33_4.mp4')

    # initialise tracker
    tracker = Tracker("models/best.pt") 

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_tranformer = ViewTransformer() 
    view_tranformer.add_transformed_position_to_tracks(tracks)


    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_postions(tracks["ball"])

    # Speed Distance Estimator
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_distance_to_tracks(tracks)

    # Assign teams
    teamassigner = TeamAssigner()
    teamassigner.assign_team_colour(video_frames[0], tracks['players'][0])

    for frame_no, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = teamassigner.get_player_team(video_frames[frame_no], track['bbox'], player_id)
            tracks['players'][frame_no][player_id]['team'] = team
            tracks['players'][frame_no][player_id]['team_color'] = teamassigner.team_colours[team]

    # Acquisiton of ball
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player, ball_bbox)  

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True   
            team_ball_control.append(tracks["players"][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    # draw speed and distance
    output_video_frames = speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # save video
    save_video(output_video_frames, 'outputs/output_video.avi')

if __name__ == '__main__':
    main()