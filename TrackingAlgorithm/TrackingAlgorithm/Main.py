import sys
from contourTrackerMainClass import contourTrackerMain

# run tracking
prg = contourTrackerMain("MainTracking.conf",runInteractive=False)
prg.initializeTracking()
prg.track()
