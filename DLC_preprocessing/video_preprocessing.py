import glob
import pandas as pd
import cv2
import numpy as np
from rembg import remove
from PIL import Image
from tqdm import tqdm

def video_clahe(input_video) :
        print (input_video.split('/')[-1])
        cap = cv2.VideoCapture(input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        N_FRAMES_OUTPUT = 20*60*fps
        frames = []
        for i in tqdm(range(N_FRAMES_OUTPUT)):
                ret, frame = cap.read()
                frames.append(frame)

        ### Contrast limited adaptive histogram equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        output_video = input_video.replace('.mp4', '_clahe.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=True)
        for frame in tqdm(frames[:N_FRAMES_OUTPUT]):
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                l, a, b = cv2.split(lab)
                l_clahe = clahe.apply(l)
                lab_clahe = cv2.merge((l_clahe, a, b))
                output_frame = cv2.cvtColor(lab_clahe, cv2.COLOR_Lab2BGR)
                out.write(output_frame)
        out.release


v_dir = '/Users/JiHyePark_1/Desktop/Jihye/CTX/DeepLabCut/rotation_videos/crop/'

#m_lst = ['9-2-1','9-2-2','9-2-3','9-3-2','9-5-1','9-5-2','9-5-3','9-6-2','9-6-3','9-6-4','9-7-1','9-7-2','9-7-3','9-7-4','9-8-1','9-8-2','9-8-4','9-9-1','9-9-2','9-9-3','9-10-1','9-10-2','9-11-1','9-12-1']

#m_lst = ['9-2-1','9-2-2','9-2-3','9-3-2','9-5-1','9-5-2','9-5-3','9-6-3','9-6-4','9-7-1','9-7-2','9-7-3','9-7-4','9-8-1','9-8-2','9-8-4','9-9-1','9-9-2','9-9-3','9-10-1','9-11-1','9-12-1']

m_lst = ['8-1-1','8-1-3','8-2-1','8-2-2','8-2-3','8-4-1','8-4-2','8-4-3','8-4-4','8-7-1','8-8-1','8-8-2','8-9-3','8-12-1','8-13-1']

#m_lst = ['7-2-2','7-6-1','7-6-2','7-6-3','7-7-4']

tp_lst = ['TP5']

print (len(m_lst))

for mouse_id in m_lst :
	for tp in tp_lst : 
		print (mouse_id)
		v_lst = glob.glob(v_dir + 'HDV_*_' + mouse_id + '-*_' + tp + '.mp4')
		print (len(v_lst))
		for v_input in v_lst :
			video_clahe(v_input)




 
