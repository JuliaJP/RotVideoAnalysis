#start in video file directory

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_rotation_count(mouse_id, scorer, v_dir) :
    #read h5 file and convert to dataframe
    #loading output of DLC
    angle_changes = []
    #clockwise_rotations = 0
    #counterclockwise_rotations = 0
    
    video = mouse_id +  '.mp4'
    dataname = str(Path(video).stem) + scorer + '.h5'
    h5_file_path = os.path.join(v_dir, dataname)
    df = pd.read_hdf(h5_file_path)
    nose = df[scorer]['nose']
    bodycentre = df[scorer]['bodycentre']
    tailbase = df[scorer]['tailbase']
    total_frames = len(nose)
    # Calculate the center of tailbase
    tailbase_center_x = tailbase['x'].mean()
    tailbase_center_y = tailbase['y'].mean()
    bodycentre_center_x = bodycentre['x'].mean()
    bodycentre_center_y = bodycentre['y'].mean()

    for i in range(1, total_frames) :
        tailbase_position = tailbase.iloc[i]
        bodycentre_position = bodycentre.iloc[i]
        nose_prev_position = nose.iloc[i-1]
        nose_current_position = nose.iloc[i]

        dx_prev = nose_prev_position['x'] - bodycentre_position['x']
        dy_prev = nose_prev_position['y'] - bodycentre_position['y']
        dx_current = nose_current_position['x'] - bodycentre_position['x']
        dy_current = nose_current_position['y'] - bodycentre_position['y']

        #dx_prev = nose_prev_position['x'] - bodycentre_center_x
        #dy_prev = nose_prev_position['y'] - bodycentre_center_y
        #dx_current = nose_current_position['x'] - bodycentre_center_x
        #dy_current = nose_current_position['y'] - bodycentre_center_y

        #dx_prev = nose_prev_position['x'] - tailbase_position['x']
        #dy_prev = nose_prev_position['y'] - tailbase_position['y']
        #dx_current = nose_current_position['x'] - tailbase_position['x']
        #dy_current = nose_current_position['y'] - tailbase_position['y']

        #dx_prev = nose_prev_position['x'] - tailbase_center_x
        #dy_prev = nose_prev_position['y'] - tailbase_center_y
        #dx_current = nose_current_position['x'] - tailbase_center_x
        #dy_current = nose_current_position['y'] - tailbase_center_y

        angle_prev = np.arctan2(dy_prev, dx_prev)
        angle_current = np.arctan2(dy_current, dx_current)
        angle_change = angle_current - angle_prev
        #Angle change normalization between -pi and pi
        angle_change = (angle_change + np.pi) % (2 * np.pi) - np.pi
        angle_changes.append(angle_change)

    angle_changes = np.array(angle_changes)
    total_angle = np.sum(angle_changes)
    rotation_count = total_angle / (2 * np.pi)
    rotation_per_frame = rotation_count / total_frames
    rotation_per_minute = rotation_per_frame * fps * 60

    return (rotation_count, rotation_per_minute, angle_changes)



v_dir = '/data/DeepLabCut/test_videos/'
#v_lst = ['HDV_5-4-1','HDV_5-4-2','HDV_5-4-4']
DLCscorer = 'DLC_resnet50_testDLCFeb10shuffle1_100000'
fps = 30
#for each mouse (1,2,3 videos)
mouse_id = 'HDV_5-4-4-3'
(rotation_count, rotation_per_minute, angle_changes) = calculate_rotation_count(mouse_id, DLCscorer, v_dir)

print('Mouse ID : ', mouse_id)
print('Mouse Rotation Count : ', rotation_count)
print('Rotations per minute : ', rotation_per_minute)



