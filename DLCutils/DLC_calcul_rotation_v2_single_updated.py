#start in video file directory

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd

def calculate_rotation_count(v_lst, scorer, v_dir, rotation_direction, fps, outlier_factor=1.5):
    # Initialize a list to store angle changes
    angle_changes = []

    for h5_file_path in v_lst:
        df = pd.read_hdf(h5_file_path)
        nose = df[scorer]['nose']
        bodycentre = df[scorer]['bodycentre']
        total_frames = len(bodycentre)

        for i in range(1, total_frames):
            bodycentre_position = bodycentre.iloc[i]
            nose_prev_position = nose.iloc[i - 1]
            nose_current_position = nose.iloc[i]

            # Calculate the displacement of the nose relative to the body centre
            dx_prev = nose_prev_position['x'] - bodycentre_position['x']
            dy_prev = nose_prev_position['y'] - bodycentre_position['y']
            dx_current = nose_current_position['x'] - bodycentre_position['x']
            dy_current = nose_current_position['y'] - bodycentre_position['y']

            # Calculate the angle of the nose relative to the body centre
            angle_prev = np.arctan2(dy_prev, dx_prev)
            angle_current = np.arctan2(dy_current, dx_current)
            angle_change = angle_current - angle_prev

            # If the rotation direction is counter-clockwise, reverse the sign of the angle change
            if rotation_direction == 'counter-clockwise':
                angle_change = -angle_change

            # Normalize the angle change to be between -pi and pi
            angle_change = (angle_change + np.pi) % (2 * np.pi) - np.pi
            angle_changes.append(angle_change)

    # Calculate mean and standard deviation of angle changes
    mean_angle_change = np.mean(angle_changes)
    std_angle_change = np.std(angle_changes)

    # Define angle change threshold for outlier detection
    angle_threshold = mean_angle_change + outlier_factor * std_angle_change

    # Exclude angle changes that are considered outliers
    angle_changes = [angle_change for angle_change in angle_changes if abs(angle_change) <= angle_threshold]

    # Calculate total rotation count
    total_angle = np.sum(angle_changes)
    rotation_count = total_angle / (2 * np.pi)

    # Calculate total recording time in seconds
    total_time_seconds = len(angle_changes) / fps

    # Calculate rotations per minute based on total recording time
    total_time_minutes = total_time_seconds / 60 if total_time_seconds > 0 else 1
    rotation_per_minute = rotation_count / total_time_minutes

    return rotation_count, rotation_per_minute


#v_dir = '/data/DeepLabCut/rotation_videos/labeled/'
v_dir = '/Users/JiHyePark_1/Desktop/DeepLabCut/rotation_videos/labeled/'

m_lst = ['J-2-1','J-2-4','J-4-3','J-5-2','J-5-3']
#m_lst = ['K-1-3','K-2-1','K-3-1','K-3-2','K-5-2','K-7-1','K-7-2','K-9-1','K-9-5','K-10-1','K-12-4','K-13-2','K-10-3']
#m_lst = ['L-3-2','L-2-1']
#m_lst = ['M-4-1','M-4-3','M-4-4','M-5-2','M-6-1','M-6-2','M-6-4','M-6-5','M-3-1','M-3-2','M-7-1','M-8-1']
#m_lst = ['N-2-1','N-2-3','N-3-3','N-4-2','N-5-2','N-8-1','N-8-2','N-8-3','N-7-1','N-7-2','N-7-3','N-7-4']
#m_lst = ['M-1-1','M-1-2','M-1-3','M-2-4','M-2-5','M-5-2','M-6-1','M-6-2','M-6-4','M-3-1','M-3-2']
#m_lst = ['K-1-3','K-2-1','K-3-1','K-3-2','K-5-2','K-9-1','K-9-5','K-10-1']
#m_lst = ['K-7-1','K-7-2','K-12-4','K-13-2','K-10-3']

tp_lst = ['TP6']
tp_mnt = 'TP 6month'

DLCscorer = 'DLC_resnet50_rottestApr27shuffle2_1000000'
fps = 30

print ('      ')
print (" Rotation test results (turns / min)")
print ('      ')
print ('+----------+-----------+-----------+')
print ('| mouse_id | ' + 'condition' + ' |   turns   |' )
for mouse_id in m_lst : 
    out_line = []
    if mouse_id.startswith('left') :
        rotation_direction = 'counter-clockwise'
    else : 
        rotation_direction = 'clockwise'
    for tp in tp_lst :
        f_lst = glob.glob(v_dir + 'HDV_*_' + mouse_id + '_' + tp + '_clahe' + DLCscorer + '.h5')
        (rot_count, rot_per_minute) = calculate_rotation_count(f_lst, DLCscorer, v_dir, rotation_direction, fps)
        out_line.append(str(round(rot_per_minute,2)))
        print ('+----------+-----------+-----------+')
        print ('| ' + mouse_id + '    | ' + tp_mnt + ' |   ' + '\t'.join(out_line) + '   |')
print ('+----------+-----------+-----------+')
