import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

v_dir = '/Users/JiHyePark_1/Desktop/Jihye/CTX/DeepLabCut/rotation_videos/labeled/'
out_dir = '/Users/JiHyePark_1/Desktop/Jihye/CTX/DeepLabCut/Ypos_plot/Ypos_nose_'
DLCscorer = 'DLC_resnet50_rottestApr27shuffle2_1000000'
fps = 30

def plot_yaxis(v_inputs, mouse_id, scorer, out_dir, bpt='nose'):
	fig, axs = plt.subplots(len(v_inputs), 1, figsize=(10, 15))  # Adjust the size as necessary

	for idx, v_input in enumerate(v_inputs):
		df = pd.read_hdf(v_input)
		tmp = v_input.split('_')
		tp = tmp[5]
		vn = tmp[4].split('-')[-1]
		frames = len(df[scorer][bpt])
		time = np.arange(frames)*1./fps/60  # 60으로 나누어 분으로 변환
		ysnout = df[scorer][bpt]['y'].values
		xsnout = df[scorer][bpt]['x'].values		

		axs[idx].plot(time, ysnout)
		axs[idx].set_xlabel('Time in minutes')
		axs[idx].set_ylabel('Y-position in pixel (nose)')
		axs[idx].set_title(mouse_id + '_' + tp)

	plt.tight_layout()
	plt.savefig(out_dir + mouse_id + '_' + tp + '.png')
	plt.close()


#m_lst = ['4-1-3','4-1-4','4-2-1','4-2-2','4-2-3','left-3','2-1','3-1-1']

#m_lst = ['8-1-1','8-1-3','8-2-1','8-2-2','8-2-3','8-4-1','8-4-2','8-4-3','8-4-4','8-5-2','8-5-3','8-7-1','8-7-2','8-8-1','8-8-2','8-9-1','8-9-3','8-12-1','8-13-1']

m_lst = ['7-2-2','7-6-1','7-6-2','7-7-4','7-6-3']

#tp_lst = ['TP0','TP1','TP2','TP3','TP4','TP5','TP6']
tp_lst = ['TP6']

for m_id in m_lst :
	for tp in tp_lst :
		f_lst = glob.glob(v_dir + 'HDV_*_' + m_id + '-*_' + tp + '_clahe' + DLCscorer + '.h5')
		plot_yaxis(f_lst, m_id, DLCscorer, out_dir)

