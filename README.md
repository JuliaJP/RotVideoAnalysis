# RotVideoAnalysis
Mouse rotation test video analysis with DeepLabCut

* Extract frames from input video
    * 15 frames from 8 randomly selected videos
* Training dataset - train network : 1,030,000 iterations
* Evaluate network
* Extract outlier frames & refine tracklets
* Analyze video
    * Place the bodycenter at the center of the circle
    * Measure the angle change of the nose for every frame
    * Total frames : 30 fps * 60 sec * 60 min (60 min video)
  

