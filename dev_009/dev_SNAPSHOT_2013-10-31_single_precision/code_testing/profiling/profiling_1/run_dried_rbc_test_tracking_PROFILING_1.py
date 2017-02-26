import sys
import ipdb
import scipy.io as io
import numpy as np

sys.path.append("/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/own/flicker_spectroscopy/image_interpolation/opencl_interpolation_testing/tracking_algorithm/dev_008/dev/")

from contourTrackerMainClass import contourTrackerMain

# run tracking
prg = contourTrackerMain("dried_rbc_test_tracking_PROFILING_1.conf")
prg.setProfiling(True)
#~ prg.setProfiling(False)
prg.initializeTracking()
prg.track()

# check results
#~ ipdb.set_trace()
dataAnalysisDirectoryReferencePath = "/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/own/flicker_spectroscopy/image_interpolation/opencl_interpolation_testing/tracking_algorithm/dev_008/dev/code_testing/profiling/profiling_1/dried_rbc_test_tracking_PROFILING_1_unit-test_reference"
tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesX.mat')
contourCoordinatesXref = tmp['contourCoordinatesX']
tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesY.mat')
contourCoordinatesYref = tmp['contourCoordinatesY']

print ""
if np.all(prg.contourCoordinatesY == contourCoordinatesYref) and \
   np.all(prg.contourCoordinatesX == contourCoordinatesXref):
	print "Unit PASSED: Tracking results are same!"
else:
	print "Unit FAILED: Tracking results are different!"
		
