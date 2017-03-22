import unittest
import sys
import scipy.io as io
import numpy as np
from contourTrackerMainClass import contourTrackerMain

class Test_integrationTests(unittest.TestCase):
	def test_CompleteProgramRun_000(self):
		sys.path.append("/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/own/flicker_spectroscopy/image_interpolation/opencl_interpolation_testing/tracking_algorithm/dev_006/dev_SNAPSHOT_2014-03-12/")

		prg = contourTrackerMain("MainTracking.conf",runInteractive=False)
		prg.initializeTracking()
		prg.track()

		dataAnalysisDirectoryReferencePath = "C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/IntegrationTests/CompleteProgramRun_000"
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesX.mat')
		contourCoordinatesXref = tmp['contourCoordinatesX']
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesY.mat')
		contourCoordinatesYref = tmp['contourCoordinatesY']
		
		self.assertTrue(np.all(prg.contourCoordinatesX == contourCoordinatesXref))
		self.assertTrue(np.all(prg.contourCoordinatesY == contourCoordinatesYref))

if __name__ == '__main__':
	unittest.main()
