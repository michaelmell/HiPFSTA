import sys
#~ sys.path.append("/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/own/flicker_spectroscopy/image_interpolation/opencl_interpolation_testing/tracking_algorithm/dev_006/dev_SNAPSHOT_2014-05-23/")
sys.path.append("C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/")

from contourTrackerMainClass import contourTrackerMain

# run tracking
prg = contourTrackerMain("tracking_000.conf",runInteractive=True)
# prg.initializeTracking()
prg.startTracking()
