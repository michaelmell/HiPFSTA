import sys
from contourTrackerMainClass import contourTrackerMain

# run tracking
prg = contourTrackerMain("TrackingConfigs/TrackingConfig_000.conf",runInteractive=False)
prg.initializeTracking()
prg.track()
