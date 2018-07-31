import sys
sys.path.append("../../../TrackingAlgorithm/TrackingAlgorithm/")
from contourTrackerMainClass import contourTrackerMain

prg = contourTrackerMain("tracking_config.conf",runInteractive=True)
prg.startTracking()