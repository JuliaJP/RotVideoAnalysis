
import glob
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

def video_clahe(input_video):
    print(input_video.split('/')[-1])
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ### Contrast limited adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    output_video = input_video.replace('.mp4', '_clahe.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=True)

    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        output_frame = cv2.cvtColor(lab_clahe, cv2.COLOR_Lab2BGR)
        out.write(output_frame)

    cap.release()
    out.release()

v_dir = '/Users/JiHyePark_1/Desktop/DeepLabCut/rotation_videos/crop/'

m_lst = ['J-2-1','J-2-4','J-4-3','J-5-2','J-5-3']
#m_lst = ['K-1-3','K-2-1','K-3-1','K-3-2','K-9-1','K-9-5','K-7-1','K-7-2','K-12-4','K-10-1','K-10-3','K-13-2','K-5-2']
#m_lst = ['L-3-2','L-2-1']
#m_lst = ['N-2-1','N-2-3','N-3-3','N-4-2','N-5-2','N-8-1','N-8-2','N-8-3','N-7-1','N-7-2','N-7-3','N-7-4']
#m_lst = ['M-1-1','M-1-2','M-1-3','M-2-4','M-2-5','M-5-2','M-6-1','M-6-2','M-6-4','M-3-1','M-3-2']
#m_lst = ['K-1-3','K-2-1','K-3-1','K-3-2','K-5-2','K-9-1','K-9-5','K-10-1']
#m_lst = ['K-7-1','K-7-2','K-12-4','K-13-2','K-10-3']


tp_lst = ['TP6']

print (len(m_lst))

for mouse_id in m_lst :
        for tp in tp_lst :
                print (mouse_id)
                v_lst = glob.glob(v_dir + 'HDV_*_' + mouse_id + '_' + tp + '.mp4')
                for v_input in v_lst :
                        video_clahe(v_input)
