import sys
sys.path.append("/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/own/flicker_spectroscopy/image_interpolation/opencl_interpolation_testing/tracking_algorithm/dev_007/dev/")

from contourTrackerMainClass import contourTrackerMain

#~ contourTrackerMain = reload(contourTrackerMain)

prg = contourTrackerMain("dried_rbc_test_tracking_2.conf")
prg.initializeTracking()
prg.track()
