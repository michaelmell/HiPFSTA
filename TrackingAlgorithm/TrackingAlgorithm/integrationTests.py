import unittest
import sys
import scipy.io as io
import numpy as np
from contourTrackerMainClass import contourTrackerMain

class Test_integrationTests(unittest.TestCase):
	_equalityTolerance=1e-16

	def test_CompleteProgramRun_000(self):
		prg = contourTrackerMain("TrackingConfigs/TrackingConfig_000.conf",runInteractive=False)
		prg.startTracking()

		dataAnalysisDirectoryReferencePath = "C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/IntegrationTests/CompleteProgramRun_000"
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesX.mat')
		contourCoordinatesXref = tmp['contourCoordinatesX']
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesY.mat')
		contourCoordinatesYref = tmp['contourCoordinatesY']
		
		self.assertTrue(np.allclose(prg.contourCoordinatesX,contourCoordinatesXref,atol=self._equalityTolerance,equal_nan=False))
		self.assertTrue(np.allclose(prg.contourCoordinatesY,contourCoordinatesYref,atol=self._equalityTolerance,equal_nan=False))

	def test_CompleteProgramRun_001(self):
		prg = contourTrackerMain("TrackingConfigs/TrackingConfig_002.conf",runInteractive=False)
		prg.startTracking()

		dataAnalysisDirectoryReferencePath = "C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/IntegrationTests/CompleteProgramRun_001"
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesX.mat')
		contourCoordinatesXref = tmp['contourCoordinatesX']
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesY.mat')
		contourCoordinatesYref = tmp['contourCoordinatesY']
		
		self.assertTrue(np.allclose(prg.contourCoordinatesX,contourCoordinatesXref,atol=self._equalityTolerance,equal_nan=False))
		self.assertTrue(np.allclose(prg.contourCoordinatesY,contourCoordinatesYref,atol=self._equalityTolerance,equal_nan=False))

	def test_CompleteProgramRun_Tracking_Continuation_000(self):
		prg = contourTrackerMain("TrackingConfigs/TrackingConfig_001.conf",runInteractive=False)
		prg.startTracking()

		dataAnalysisDirectoryReferencePath = "C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/IntegrationTests/CompleteProgramRun_000"
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesX.mat')
		contourCoordinatesXref = tmp['contourCoordinatesX']
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesY.mat')
		contourCoordinatesYref = tmp['contourCoordinatesY']
		
		self.assertTrue(np.allclose(prg.contourCoordinatesX,contourCoordinatesXref,atol=self._equalityTolerance,equal_nan=False))
		self.assertTrue(np.allclose(prg.contourCoordinatesY,contourCoordinatesYref,atol=self._equalityTolerance,equal_nan=False))

	def test_CompleteProgramRun_Tracking_Continuation_001(self):
		prg = contourTrackerMain("TrackingConfigs/TrackingConfig_003.conf",runInteractive=False)
		prg.startTracking()

		dataAnalysisDirectoryReferencePath = "C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/IntegrationTests/CompleteProgramRun_001"
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesX.mat')
		contourCoordinatesXref = tmp['contourCoordinatesX']
		tmp=io.loadmat(dataAnalysisDirectoryReferencePath+'/contourCoordinatesY.mat')
		contourCoordinatesYref = tmp['contourCoordinatesY']
		
		self.assertTrue(np.allclose(prg.contourCoordinatesX,contourCoordinatesXref,atol=self._equalityTolerance,equal_nan=False))
		self.assertTrue(np.allclose(prg.contourCoordinatesY,contourCoordinatesYref,atol=self._equalityTolerance,equal_nan=False))

if __name__ == '__main__':
	unittest.main()
