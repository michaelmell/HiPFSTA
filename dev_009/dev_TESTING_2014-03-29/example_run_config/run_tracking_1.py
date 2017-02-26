import sys
sys.path.append("/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/own/flicker_spectroscopy/image_interpolation/opencl_interpolation_testing/tracking_algorithm/dev_009/dev_TESTING_2014-03-29/")

from contourTrackerMainClass import contourTrackerMain

# run tracking
prg = contourTrackerMain("tracking_1.conf",runInteractive=False)
prg.initializeTracking()
prg.track()
