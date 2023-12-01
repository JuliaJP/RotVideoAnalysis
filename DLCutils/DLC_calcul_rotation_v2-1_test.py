#start in video file directory

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd

def calculate_rotation_count(v_lst, scorer, v_dir, rotation_direction, continuous_threshold=10, angle_change_threshold=0.01, outlier_factor=1.5) :
    #read h5 file and convert to dataframe
    #loading output of DLC
    angle_changes = []
    continuous_count = 0
	
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
           
            # If angle change is above the threshold and continuous in the same direction, increment the continuous count
            if abs(angle_change) >= angle_change_threshold and (len(angle_changes) == 0 or np.sign(angle_change) == np.sign(angle_changes[-1])):
                continuous_count += 1
            else:
                continuous_count = 0

            # If the continuous count exceeds the threshold, append the angle change
            if continuous_count >= continuous_threshold:
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
#m_lst = ['4-2-4','left-4']
#m_lst = ['4-1-3','4-2-1','4-2-2','4-2-3','left-3','1-3','2-1','3-1-1','3-2-1','3-2-3']
#m_lst = ['7-2-2','7-4-2','7-7-4','7-6-1','7-6-2','7-6-3']
m_lst = ['8-5-2']
#tp_lst = ['TP0','TP1','TP2','TP3','TP4','TP5','TP6']
tp_lst = ['TP3']
DLCscorer = 'DLC_resnet50_rottestApr27shuffle2_1000000'
fps = 30

print ('mouse_id\t' + '\t'.join(tp_lst) )
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
    print (mouse_id + '\t' + '\t'.join(out_line) )

