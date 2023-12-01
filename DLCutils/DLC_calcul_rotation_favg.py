#start in video file directory

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_rotation_count(mouse_id, scorer, v_dir, avg_window=3) :
    #read h5 file and convert to dataframe
    #loading output of DLC
    angle_changes = []
    clockwise_rotations = 0
    counterclockwise_rotations = 0
    whole_frames = 0

    for x in range(1, 4) :
        video = mouse_id + '-' + str(x) + '.mp4'
        dataname = str(Path(video).stem) + scorer + '.h5'
        h5_file_path = os.path.join(v_dir, dataname)
        df = pd.read_hdf(h5_file_path)
        nose = df[scorer]['nose']
        tailbase = df[scorer]['tailbase']
        total_frames = len(nose)
        whole_frames = whole_frames + total_frames

        for i in range(avg_window, len(nose), avg_window) :
            angle_changes_window = []
            for j in range(avg_window) :
                tailbase_position = tailbase.iloc[i-j]
                nose_prev_position = nose.iloc[i-j-1]
                nose_current_position = nose.iloc[i-j]

                dx_prev = nose_prev_position['x'] - tailbase_position['x']
                dy_prev = nose_prev_position['y'] - tailbase_position['y']
                dx_current = nose_current_position['x'] - tailbase_position['x']
                dy_current = nose_current_position['y'] - tailbase_position['y']

                angle_prev = np.arctan2(dy_prev, dx_prev)
                angle_current = np.arctan2(dy_current, dx_current)
                angle_change = angle_current - angle_prev
                #Angle change normalization between -pi and pi
                angle_change = (angle_change + np.pi) % (2 * np.pi) - np.pi
                angle_changes_window.append(angle_change)
            
            avg_angle_change = np.mean(angle_changes_window)
            angle_changes.append(avg_angle_change)

    angle_changes = np.array(angle_changes)
    total_angle = np.sum(angle_changes)
    rotation_count = total_angle / (2 * np.pi)
    total_seconds = whole_frames / fps 
    rotation_per_minute = rotation_count * (60/total_seconds)

    return (rotation_count, rotation_per_minute, angle_changes)



v_dir = '/data/DeepLabCut/test_videos/'
#v_lst = ['HDV_5-4-1','HDV_5-4-2','HDV_5-4-4']
DLCscorer = 'DLC_resnet50_testDLCFeb10shuffle1_100000'
fps = 30
mouse_id = 'HDV_5-4-4'
(rotation_count, rotation_per_minute, angle_changes) = calculate_rotation_count(mouse_id, DLCscorer, v_dir)

print('Mouse ID : ', mouse_id)
print('Mouse Rotation Count : ', rotation_count)
print('Rotations per minute : ', rotation_per_minute)
