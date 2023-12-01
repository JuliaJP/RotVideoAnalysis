#start in video file directory

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd

def calculate_rotation_count(v_lst, scorer, v_dir, rotation_direction, outlier_factor=1.5) :
    #read h5 file and convert to dataframe
    #loading output of DLC
    angle_changes = []

    for h5_file_path in v_lst : 
        df = pd.read_hdf(h5_file_path)
        nose = df[scorer]['nose']
        bodycentre = df[scorer]['bodycentre']
        total_frames = len(bodycentre)

        for i in range(1, total_frames) :
            bodycentre_position = bodycentre.iloc[i]
            nose_prev_position = nose.iloc[i-1]
            nose_current_position = nose.iloc[i]

            dx_prev = nose_prev_position['x'] - bodycentre_position['x']
            dy_prev = nose_prev_position['y'] - bodycentre_position['y']
            dx_current = nose_current_position['x'] - bodycentre_position['x']
            dy_current = nose_current_position['y'] - bodycentre_position['y']

            angle_prev = np.arctan2(dy_prev, dx_prev)
            angle_current = np.arctan2(dy_current, dx_current)
            angle_change = angle_current - angle_prev

            # If rotation direction is counter-clockwise, change sign of angle change
            if rotation_direction == 'counter-clockwise':
                angle_change = -angle_change

            angle_change = (angle_change + np.pi) % (2 * np.pi) - np.pi
            angle_changes.append(angle_change)

    # Calculate mean and standard deviation of angle changes
    mean_angle_change = np.mean(angle_changes)
    std_angle_change = np.std(angle_changes)

    # Define angle change threshold for outlier detection
    angle_threshold = mean_angle_change + outlier_factor * std_angle_change

    # Exclude angle changes that are considered outliers
    angle_changes = [angle_change for angle_change in angle_changes if abs(angle_change) <= angle_threshold]

    total_angle = np.sum(angle_changes)
    rotation_count = total_angle / (2 * np.pi)
    rotation_per_frame = rotation_count / len(angle_changes) 
    rotation_per_minute = rotation_per_frame * fps * 60

    return (rotation_count, rotation_per_minute)


#v_dir = '/data/DeepLabCut/rotation_videos/labeled/'
v_dir = '/Users/JiHyePark_1/Desktop/Jihye/CTX/DeepLabCut/rotation_videos/labeled/'

#m_lst = ['7-2-2','7-6-1','7-6-2','7-7-4','7-6-3']

m_lst = ['8-1-1','8-1-3','8-2-1','8-2-2','8-2-3','8-4-1','8-4-2','8-4-3','8-4-4','8-5-3','8-7-1','8-7-2','8-8-1','8-8-2','8-9-3','8-12-1','8-13-1']

#m_lst = ['9-2-1','9-2-2','9-2-3','9-3-2','9-5-1','9-5-2','9-5-3','9-6-3','9-6-4','9-7-1','9-7-2','9-7-3','9-7-4','9-8-1','9-8-2','9-8-4','9-9-1','9-9-2','9-9-3','9-10-1','9-11-1','9-12-1']

tp_lst = ['TP4']
tp_mnt = 'TP 4month'

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
        f_lst = glob.glob(v_dir + 'HDV_*_' + mouse_id + '-*_' + tp + '_clahe' + DLCscorer + '.h5')
        (rot_count, rot_per_minute) = calculate_rotation_count(f_lst, DLCscorer, v_dir, rotation_direction)
        out_line.append(str(round(rot_per_minute,2)))
        print ('+----------+-----------+-----------+')
        print ('| ' + mouse_id + '    | ' + tp_mnt + ' |   ' + '\t'.join(out_line) + '   |')
print ('+----------+-----------+-----------+')
