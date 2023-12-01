#start in video file directory

import time_in_each_roi
import pandas as pd
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt

v_dir = '/data/DeepLabCut/test_videos/'
video='HDV_0535_1.mp4'
DLCscorer = 'DLC_resnet50_testDLCFeb10shuffle1_100000'
dataname = str(Path(video).stem) + DLCscorer + '.h5'
#loading output of DLC
Datafi= pd.read_hdf(v_dir + os.path.join(dataname))
#these structures are awesome to manipulate, how -->> see pandas https://pandas.pydata.org/pandas-docs/stable/index.html
Dataframe.head()

#bodyparts=Dataframe.columns.get_level_values(1) bodyparts2plot=bodyparts
#%matplotlib inline
#DLC_pm.PlottingResults(Dataframe,bodyparts2plot,alphavalue=.2,pcutoff=.5,fs=(8,4))

#let's calculate velocity of the snout
bpt = 'nose'
vel = time_in_each_roi.calc_distance_between_points_in_a_vector_2d(np.vstack([Dataframe[DLCscorer][bpt]['x'].values.flatten(), Dataframe[DLCscorer][bpt]['y'].values.flatten()]).T)

fps=30 #frame rate of camera in those experiments 
time=np.arange(len(vel))*1./fps
vel=vel #notice the units of vel are relative pixel distance [per time step]

xsnout=Dataframe[DLCscorer][bpt]['x'].values
ysnout=Dataframe[DLCscorer][bpt]['y'].values
vsnout=vel

#%matplotlib inline

plt.plot(time,vel*1./fps)
plt.xlabel('Time in seconds')
plt.ylabel('Speed in pixels per second')
plt.show()


