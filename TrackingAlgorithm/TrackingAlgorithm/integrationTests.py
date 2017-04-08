import unittest
import sys
import scipy.io as io
import numpy as np
from contourTrackerMainClass import contourTrackerMain

class Test_integrationTests(unittest.TestCase):
	def test_CompleteProgramRun_000(self):
		prg = contourTrackerMain("TrackingConfigs/TrackingConfig_000.conf",runInteractive=False)
		prg.startTracking()

		dataAnalysisDirectoryReferencePath = "C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/IntegrationTests/CompleteProgramRun_000"
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesX.mat')
		contourCoordinatesXref = tmp['contourCoordinatesX']
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesY.mat')
		contourCoordinatesYref = tmp['contourCoordinatesY']
		
		self.assertTrue(np.all(prg.contourCoordinatesX == contourCoordinatesXref))
		self.assertTrue(np.all(prg.contourCoordinatesY == contourCoordinatesYref))

	def test_CompleteProgramRun_Tracking_Continuation_000(self):
		prg = contourTrackerMain("TrackingConfigs/TrackingConfig_001.conf",runInteractive=False)
		prg.startTracking()

		dataAnalysisDirectoryReferencePath = "C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/IntegrationTests/CompleteProgramRun_000"
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesX.mat')
		contourCoordinatesXref = tmp['contourCoordinatesX']
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesY.mat')
		contourCoordinatesYref = tmp['contourCoordinatesY']
		
		self.assertTrue(np.all(prg.contourCoordinatesX == contourCoordinatesXref))
		self.assertTrue(np.all(prg.contourCoordinatesY == contourCoordinatesYref))

if __name__ == '__main__':
	unittest.main()
