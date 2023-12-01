import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

v_dir = '/Users/JiHyePark_1/Desktop/Jihye/CTX/DeepLabCut/rotation_videos/labeled/'
out_dir = '/Users/JiHyePark_1/Desktop/Jihye/CTX/DeepLabCut/Ypos_plot/Ypos_nose_'
DLCscorer = 'DLC_resnet50_rottestApr27shuffle2_1000000'
fps = 30

def plot_yaxis(v_input, mouse_id, scorer, out_dir, bpt='nose'):
	df = pd.read_hdf(v_input)
	tmp = v_input.split('_')
	tp = tmp[5]
	vn = tmp[4].split('-')[-1]
	frames = len(df[scorer][bpt])
	time = np.arange(frames)*1./fps/60
	ysnout = df[scorer][bpt]['y'].values
	xsnout = df[scorer][bpt]['x'].values

	plt.figure()
	plt.plot(time, ysnout)
	plt.xlabel('Time in minutes')
	plt.ylabel('Y-position in pixel (nose)')
	plt.title( mouse_id + '_' + tp )
	plt.savefig( out_dir + mouse_id + '-' + vn + '_' + tp + '.png') 
	plt.close()

#m_lst = ['4-2-4','left-4','4-1-3','4-2-1','4-2-2','4-2-3','left-3','1-3','2-1','3-1-1','3-2-1','3-2-3','7-2-2','7-4-2','7-7-4','7-6-1','7-6-2','7-6-3','5-4-1']

m_lst = ['6-3-3']

for m_id in m_lst :
	f_lst = glob.glob(v_dir + 'HDV_*_' + m_id + '-*' + '_clahe' + DLCscorer + '.h5')	  
	#tp_lst = list(set([ i.split('_')[5] for i in f_lst ]))
	for f_input in f_lst :
		plot_yaxis(f_input, m_id, DLCscorer, out_dir)
		


