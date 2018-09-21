#################################################################### 
### Runs HiPFSTA to track contours in the GUV dataset.
#################################################################### 
import sys
sys.path.append("../../../TrackingAlgorithm/TrackingAlgorithm/")
from contourTrackerMainClass import contourTrackerMain

prg = contourTrackerMain("python_tracking.conf",runInteractive=True)
prg.startTracking()
