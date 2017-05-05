import unittest
import os
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import matplotlib.pyplot as plt
from mako.template import Template

class Test_testOpenClKernels(unittest.TestCase):
	_equalityTolerance=1e-16
	
	def test_findMembranePositionUsingMaxIncline(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "maximumIntensityIncline"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)

		inputPath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/findMembranePositionUsingMaxIncline_000/input'
		referencePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/findMembranePositionUsingMaxIncline_000/output'
		referenceVariableName1 = 'dev_membraneCoordinates'
		referenceVariableName2 = 'dev_membraneNormalVectors'
		referenceVariableName3 = 'dev_fitInclines'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)
		self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)

		self.loadHostVariable('trackingGlobalSize',inputPath)
		self.loadHostVariable('trackingWorkGroupSize',inputPath)
		self.loadHostVariable('host_Img',inputPath)
		self.dev_Img = cl.image_from_array(self.ctx, ary=self.host_Img, mode="r", norm_int=False, num_channels=1)
		self.loadHostVariable('imgSizeX',inputPath)
		self.loadHostVariable('imgSizeY',inputPath)
		#self.saveDeviceVariable('buf_localRotationMatrices',inputPath)
		self.loadHostVariable('localRotationMatrices',inputPath)
		self.buf_localRotationMatrices = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.localRotationMatrices)
		#self.saveDeviceVariable('buf_linFitSearchRangeXvalues',inputPath)
		self.loadHostVariable('linFitSearchRangeXvalues',inputPath)
		self.buf_linFitSearchRangeXvalues = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.linFitSearchRangeXvalues)
		self.loadHostVariable('linFitParameter',inputPath)
		self.loadHostVariable('fitIntercept_memSize',inputPath)
		self.fitIncline_memSize = self.fitIntercept_memSize
		self.loadHostVariable('rotatedUnitVector_memSize',inputPath)
		self.loadHostVariable('meanParameter',inputPath)
		#self.saveDeviceVariable('buf_meanRangeXvalues',inputPath)
		self.loadHostVariable('meanRangeXvalues',inputPath)
		self.buf_meanRangeXvalues = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.meanRangeXvalues)
		self.loadHostVariable('meanRangePositionOffset',inputPath)
		self.loadHostVariable('localMembranePositions_memSize',inputPath)
		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_membraneNormalVectors',inputPath)
		self.loadDeviceVariable('dev_fitInclines',inputPath)
		self.loadHostVariable('inclineTolerance',inputPath)
		self.inclineRefinementRange = np.int32(2)
		self.setWorkGroupSizes()

		for strideNr in range(self.nrOfStrides):
			# set the starting index of the coordinate array for each kernel instance
			kernelCoordinateStartingIndex = np.int32(strideNr*self.detectionKernelStrideSize)

			self.prg.findMembranePosition(self.queue, self.trackingGlobalSize, self.trackingWorkGroupSize, self.sampler, \
											self.dev_Img, self.imgSizeX, self.imgSizeY, \
											self.buf_localRotationMatrices, \
											self.buf_linFitSearchRangeXvalues, \
											self.linFitParameter, \
											cl.LocalMemory(self.fitIntercept_memSize), cl.LocalMemory(self.fitIncline_memSize), \
											cl.LocalMemory(self.rotatedUnitVector_memSize), \
											self.meanParameter, \
											self.buf_meanRangeXvalues, self.meanRangePositionOffset, \
											cl.LocalMemory(self.localMembranePositions_memSize), \
											self.dev_membraneCoordinates.data, \
											self.dev_membraneNormalVectors.data, \
											self.dev_fitInclines.data, \
											kernelCoordinateStartingIndex, \
											self.inclineTolerance, \
											self.inclineRefinementRange )

		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVector2EqualsExpectedResult(self.dev_membraneCoordinates,referencePath+'/'+referenceVariableName1+'.npy')
		self.assertVector2EqualsExpectedResult(self.dev_membraneNormalVectors,referencePath+'/'+referenceVariableName2+'.npy')
		self.assertVectorEqualsExpectedResult(self.dev_fitInclines,referencePath+'/'+referenceVariableName3+'.npy')
		pass

	def test_findMembranePosition(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)

		inputPath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/findMembranePosition_000/input'
		referencePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/findMembranePosition_000/output'
		referenceVariableName1 = 'dev_membraneCoordinates'
		referenceVariableName2 = 'dev_membraneNormalVectors'
		referenceVariableName3 = 'dev_fitInclines'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)
		self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)

		self.loadHostVariable('trackingGlobalSize',inputPath)
		self.loadHostVariable('trackingWorkGroupSize',inputPath)
		self.loadHostVariable('host_Img',inputPath)
		self.dev_Img = cl.image_from_array(self.ctx, ary=self.host_Img, mode="r", norm_int=False, num_channels=1)
		self.loadHostVariable('imgSizeX',inputPath)
		self.loadHostVariable('imgSizeY',inputPath)
		#self.saveDeviceVariable('buf_localRotationMatrices',inputPath)
		self.loadHostVariable('localRotationMatrices',inputPath)
		self.buf_localRotationMatrices = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.localRotationMatrices)
		#self.saveDeviceVariable('buf_linFitSearchRangeXvalues',inputPath)
		self.loadHostVariable('linFitSearchRangeXvalues',inputPath)
		self.buf_linFitSearchRangeXvalues = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.linFitSearchRangeXvalues)
		self.loadHostVariable('linFitParameter',inputPath)
		self.loadHostVariable('fitIntercept_memSize',inputPath)
		self.fitIncline_memSize = self.fitIntercept_memSize
		self.loadHostVariable('rotatedUnitVector_memSize',inputPath)
		self.loadHostVariable('meanParameter',inputPath)
		#self.saveDeviceVariable('buf_meanRangeXvalues',inputPath)
		self.loadHostVariable('meanRangeXvalues',inputPath)
		self.buf_meanRangeXvalues = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.meanRangeXvalues)
		self.loadHostVariable('meanRangePositionOffset',inputPath)
		self.loadHostVariable('localMembranePositions_memSize',inputPath)
		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_membraneNormalVectors',inputPath)
		self.loadDeviceVariable('dev_fitInclines',inputPath)
		self.loadHostVariable('inclineTolerance',inputPath)
		self.inclineRefinementRange = np.int32(2)
		self.setWorkGroupSizes()

		for strideNr in range(self.nrOfStrides):
			# set the starting index of the coordinate array for each kernel instance
			kernelCoordinateStartingIndex = np.int32(strideNr*self.detectionKernelStrideSize)

			self.prg.findMembranePosition(self.queue, self.trackingGlobalSize, self.trackingWorkGroupSize, self.sampler, \
												self.dev_Img, self.imgSizeX, self.imgSizeY, \
												self.buf_localRotationMatrices, \
												self.buf_linFitSearchRangeXvalues, \
												self.linFitParameter, \
												cl.LocalMemory(self.fitIntercept_memSize), cl.LocalMemory(self.fitIncline_memSize), \
												cl.LocalMemory(self.rotatedUnitVector_memSize), \
												self.meanParameter, \
												self.buf_meanRangeXvalues, self.meanRangePositionOffset, \
												cl.LocalMemory(self.localMembranePositions_memSize), \
												self.dev_membraneCoordinates.data, \
												self.dev_membraneNormalVectors.data, \
												self.dev_fitInclines.data, \
												kernelCoordinateStartingIndex, \
												self.inclineTolerance, \
												self.inclineRefinementRange)

		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVector2EqualsExpectedResult(self.dev_membraneCoordinates,referencePath+'/'+referenceVariableName1+'.npy')
		self.assertVector2EqualsExpectedResult(self.dev_membraneNormalVectors,referencePath+'/'+referenceVariableName2+'.npy')
		self.assertVectorEqualsExpectedResult(self.dev_fitInclines,referencePath+'/'+referenceVariableName3+'.npy')
		pass

	def test_filterNanValues_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/filterNanValues_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_membraneCoordinates'
		referenceVariableName2 = 'dev_membraneNormalVectors'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_membraneNormalVectors',inputPath)
		self.loadDeviceVariable('dev_closestLowerNoneNanIndex',inputPath)
		self.loadDeviceVariable('dev_closestUpperNoneNanIndex',inputPath)

		self.setWorkGroupSizes()

		self.prg.filterNanValues(self.queue, self.gradientGlobalSize, None, \
								 self.dev_membraneCoordinates.data, \
								 self.dev_membraneNormalVectors.data, \
								 cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes) \
								 )
		barrierEvent = cl.enqueue_barrier(self.queue)
		self.assertVector2EqualsExpectedResult(self.dev_membraneCoordinates,referencePath+'/'+referenceVariableName1+'.npy')
		self.assertVector2EqualsExpectedResult(self.dev_membraneNormalVectors,referencePath+'/'+referenceVariableName2+'.npy')
		pass

	def test_filterJumpedCoordinates_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/filterJumpedCoordinates_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_membraneCoordinates'
		referenceVariableName2 = 'dev_membraneNormalVectors'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_previousContourCenter',inputPath)
		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_membraneNormalVectors',inputPath)
		self.loadDeviceVariable('dev_previousInterpolatedMembraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_closestLowerNoneNanIndex',inputPath)
		self.loadDeviceVariable('dev_closestUpperNoneNanIndex',inputPath)
		self.maxCoordinateShift = np.float64(10.0)
		self.listOfGoodCoordinates_memSize = np.int(8192)

		self.setWorkGroupSizes()

		self.prg.filterJumpedCoordinates(self.queue, self.gradientGlobalSize, None, \
											self.dev_previousContourCenter.data, \
											self.dev_membraneCoordinates.data, \
											self.dev_membraneNormalVectors.data, \
										    self.dev_previousInterpolatedMembraneCoordinates.data, \
										    cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), \
											cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
											cl.LocalMemory(self.listOfGoodCoordinates_memSize), \
											self.maxCoordinateShift \
											)
		barrierEvent = cl.enqueue_barrier(self.queue)
		self.assertVector2EqualsExpectedResult(self.dev_membraneCoordinates,referencePath+'/'+referenceVariableName1+'.npy')
		self.assertVector2EqualsExpectedResult(self.dev_membraneNormalVectors,referencePath+'/'+referenceVariableName2+'.npy')
		pass

	def test_calculateInterCoordinateAngles_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/calculateInterCoordinateAngles_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_interCoordinateAngles'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_interCoordinateAngles',inputPath)
		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.setWorkGroupSizes()

		self.prg.calculateInterCoordinateAngles(self.queue, self.gradientGlobalSize, None, \
												self.dev_interCoordinateAngles.data, \
												self.dev_membraneCoordinates.data \
											   )
		barrierEvent = cl.enqueue_barrier(self.queue)
		self.assertVectorEqualsExpectedResult(self.dev_interCoordinateAngles,referencePath+'/'+referenceVariableName1+'.npy')
		pass

	def test_filterIncorrectCoordinates_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/filterIncorrectCoordinates_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_membraneCoordinates'
		referenceVariableName2 = 'dev_membraneNormalVectors'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_previousContourCenter',inputPath)
		self.loadDeviceVariable('dev_interCoordinateAngles',inputPath)
		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_membraneNormalVectors',inputPath)
		self.loadDeviceVariable('dev_closestLowerNoneNanIndex',inputPath)
		self.loadDeviceVariable('dev_closestUpperNoneNanIndex',inputPath)
		self.maxInterCoordinateAngle = np.float64(1.5707899999999999)
		self.setWorkGroupSizes()

		self.prg.filterIncorrectCoordinates(self.queue, self.gradientGlobalSize, None, \
											self.dev_previousContourCenter.data, \
										    self.dev_interCoordinateAngles.data, \
										    self.dev_membraneCoordinates.data, \
										    self.dev_membraneNormalVectors.data, \
										    cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
										    self.maxInterCoordinateAngle \
										    )
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVector2EqualsExpectedResult(self.dev_membraneCoordinates,referencePath+'/'+referenceVariableName1+'.npy')
		self.assertVector2EqualsExpectedResult(self.dev_membraneNormalVectors,referencePath+'/'+referenceVariableName2+'.npy')
		pass

	def test_calculateDs_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/calculateDs_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_ds'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_ds',inputPath)
		self.setWorkGroupSizes()

		self.prg.calculateDs(self.queue, self.gradientGlobalSize, None, \
					   self.dev_membraneCoordinates.data, \
					   self.dev_ds.data \
					 )
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVectorEqualsExpectedResult(self.dev_ds,referencePath+'/'+referenceVariableName1+'.npy')
		pass

	def test_calculateSumDs_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/calculateSumDs_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_sumds'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_ds',inputPath)
		self.loadDeviceVariable('dev_sumds',inputPath)
		self.setWorkGroupSizes()

		self.prg.calculateSumDs(self.queue, self.gradientGlobalSize, None, \
					   self.dev_ds.data, self.dev_sumds.data \
					 )
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVectorEqualsExpectedResult(self.dev_sumds,referencePath+'/'+referenceVariableName1+'.npy')
		pass

	def test_calculateContourCenter_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/calculateContourCenter_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_contourCenter'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_ds',inputPath)
		self.loadDeviceVariable('dev_sumds',inputPath)
		self.loadDeviceVariable('dev_contourCenter',inputPath)
		self.nrOfDetectionAngleSteps = 2048
		self.setWorkGroupSizes()

		self.prg.calculateContourCenter(self.queue, (1,1), None, \
								   self.dev_membraneCoordinates.data, \
								   self.dev_ds.data, self.dev_sumds.data, \
								   self.dev_contourCenter.data, \
								   np.int32(self.nrOfDetectionAngleSteps) \
								  )
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVector2EqualsExpectedResult(self.dev_contourCenter,referencePath+'/'+referenceVariableName1+'.npy')
		pass

	def test_cart2pol_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/cart2pol_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_membraneCoordinates'
		referenceVariableName2 = 'dev_membranePolarCoordinates'
		referenceVariableName3 = 'dev_contourCenter'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_membranePolarCoordinates',inputPath)
		self.loadDeviceVariable('dev_contourCenter',inputPath)
		self.nrOfDetectionAngleSteps = 2048
		self.setWorkGroupSizes()

		self.prg.cart2pol(self.queue, self.gradientGlobalSize, None, \
						  self.dev_membraneCoordinates.data, \
						  self.dev_membranePolarCoordinates.data, \
						  self.dev_contourCenter.data)
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVector2EqualsExpectedResult(self.dev_membraneCoordinates,referencePath+'/'+referenceVariableName1+'.npy')
		self.assertVector2EqualsExpectedResult(self.dev_membranePolarCoordinates,referencePath+'/'+referenceVariableName2+'.npy')
		self.assertVector2EqualsExpectedResult(self.dev_contourCenter,referencePath+'/'+referenceVariableName3+'.npy')
		pass

	def test_sortCoordinates_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/sortCoordinates_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_membranePolarCoordinates'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_membranePolarCoordinates',inputPath)
		self.loadDeviceVariable('dev_membraneNormalVectors',inputPath)
		self.setWorkGroupSizes()

		self.prg.sortCoordinates(self.queue, (1,1), None, \
								self.dev_membranePolarCoordinates.data, \
								self.dev_membraneCoordinates.data, \
								self.dev_membraneNormalVectors.data, \
								np.int32(self.nrOfDetectionAngleSteps) \
								)
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVector2EqualsExpectedResult(self.dev_membranePolarCoordinates,referencePath+'/'+referenceVariableName1+'.npy')
		pass

	def test_interpolatePolarCoordinatesLinear_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/interpolatePolarCoordinatesLinear_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_interpolatedMembraneCoordinates'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_membranePolarCoordinates',inputPath)
		self.loadDeviceVariable('dev_radialVectors',inputPath)
		self.loadDeviceVariable('dev_contourCenter',inputPath)
		self.loadDeviceVariable('dev_membraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_interpolatedMembraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_interpolationAngles',inputPath)
		self.nrOfAnglesToCompare=np.int32(100)
		self.setWorkGroupSizes()

		self.prg.interpolatePolarCoordinatesLinear(self.queue, self.gradientGlobalSize, None, \
													self.dev_membranePolarCoordinates.data, \
													self.dev_radialVectors.data, \
													self.dev_contourCenter.data, \
													self.dev_membraneCoordinates.data, \
													self.dev_interpolatedMembraneCoordinates.data, \
													self.dev_interpolationAngles.data, \
													self.nrOfAnglesToCompare \
													)
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVector2EqualsExpectedResult(self.dev_interpolatedMembraneCoordinates,referencePath+'/'+referenceVariableName1+'.npy')
		pass

	def test_checkIfTrackingFinished_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/checkIfTrackingFinished_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_trackingFinished'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_interpolatedMembraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_previousInterpolatedMembraneCoordinates',inputPath)
		self.loadDeviceVariable('dev_trackingFinished',inputPath)
		self.coordinateTolerance=np.float64(0.01)
		self.setWorkGroupSizes()

		self.prg.checkIfTrackingFinished(self.queue, self.gradientGlobalSize, None, \
										 self.dev_interpolatedMembraneCoordinates.data, \
										 self.dev_previousInterpolatedMembraneCoordinates.data, \
										 self.dev_trackingFinished.data, \
										 self.coordinateTolerance)
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVectorEqualsExpectedResult(self.dev_trackingFinished,referencePath+'/'+referenceVariableName1+'.npy')
		pass

	def test_checkIfCenterConverged_000(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.positioningMethod = "meanIntensityIntercept"
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/checkIfCenterConverged_000'
		inputPath = basePath+'/input'
		referencePath = basePath+'/output'
		referenceVariableName1 = 'dev_trackingFinished'

		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)

		self.loadDeviceVariable('dev_contourCenter',inputPath)
		self.loadDeviceVariable('dev_previousContourCenter',inputPath)
		self.loadDeviceVariable('dev_trackingFinished',inputPath)
		self.centerTolerance = np.float64(0.01)
		self.setWorkGroupSizes()

		self.prg.checkIfCenterConverged(self.queue, (1,1), None, \
										self.dev_contourCenter.data, \
										self.dev_previousContourCenter.data, \
										self.dev_trackingFinished.data, \
										self.centerTolerance)
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.assertVectorEqualsExpectedResult(self.dev_trackingFinished,referencePath+'/'+referenceVariableName1+'.npy')
		pass

	def assertVectorEqualsExpectedResult(self,variable,referencePath):
		outputValue = variable.get(self.queue)
		referenceValue = np.load(referencePath)
		self.assertTrue(np.allclose(outputValue, referenceValue,atol=self._equalityTolerance, equal_nan=False))

	def assertVector2EqualsExpectedResult(self,variable,referencePath):
		outputValue = variable.get(self.queue)
		referenceValue = np.load(referencePath)
		self.assertTrue(np.allclose(outputValue['x'], referenceValue['x'],atol=self._equalityTolerance, equal_nan=False))
		self.assertTrue(np.allclose(outputValue['y'], referenceValue['y'],atol=self._equalityTolerance, equal_nan=False))

	def loadHostVariable(self,variableName,path):
		host_tmp = np.load(path+'/'+variableName+'.npy')
		setattr(self,variableName,host_tmp)
		pass
	
	def loadDeviceVariable(self,variableName,path):
		host_tmp = np.load(path+'/'+variableName+'.npy')
		dev_tmp = cl_array.to_device(self.queue, host_tmp)
		setattr(self,variableName,dev_tmp)
		pass
	
	def setWorkGroupSizes(self):
		self.global_size = (1,int(self.nrOfLocalAngleSteps))
		self.local_size = (1,int(self.nrOfLocalAngleSteps))
		self.gradientGlobalSize = (int(self.nrOfDetectionAngleSteps),1)
		
		#~ ipdb.set_trace()
		#~ self.queue.device.vendor
		vendorString = self.queue.device.vendor
		# set work dimension of work group used in tracking kernel
		#~ if "intel" in vendorString.lower():  # work-around since the 'max_work_group_size' is not reported correctly for Intel-CPU using the AMD OpenCL driver (tested on: Intel(R) Core(TM) i5-3470)
		if "intel" in vendorString.lower() or "nvidia" in vendorString.lower():  # work-around since the 'max_work_group_size' is not reported correctly for Intel-CPU using the AMD OpenCL driver (tested on: Intel(R) Core(TM) i5-3470)
			self.contourPointsPerWorkGroup = 256/self.nrOfLocalAngleSteps
		else:
			self.contourPointsPerWorkGroup = self.queue.device.max_work_group_size/self.nrOfLocalAngleSteps
		
		self.trackingWorkGroupSize = (int(self.contourPointsPerWorkGroup),int(self.nrOfLocalAngleSteps))
		self.trackingGlobalSize = (int(self.detectionKernelStrideSize),int(self.nrOfLocalAngleSteps))
		
	def loadClKernels(self):
		modulePath = __file__
		codePath, filename = os.path.split(modulePath) # get location of path where our tracking code is located
		clCodeFile = codePath+"/"+"clKernelCode.cl"
		fObj = open(clCodeFile, 'r')
		self.kernelString = "".join(fObj.readlines())
		self.applyTemplating()

		text_file = open(codePath+"/"+"clKernelCode_RENDERED.cl", "w")
		text_file.write(self.kernelString)
		text_file.close()

		self.prg = cl.Program(self.ctx,self.kernelString).build()
		pass
	
	def applyTemplating(self):
		tpl = Template(self.kernelString)
		if self.positioningMethod == "maximumIntensityIncline":
			linear_fit_search_method="MAX_INCLINE_SEARCH"
		if self.positioningMethod == "meanIntensityIntercept":
			linear_fit_search_method="MIN_MAX_INTENSITY_SEARCH"
		rendered_tpl = tpl.render(linear_fit_search_method=linear_fit_search_method)
		self.kernelString=str(rendered_tpl)
		pass

	def setupClQueue(self,ctx):
		#~ self.ctx = cl.ctx([device])
		self.ctx = ctx
		self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
		self.mf = cl.mem_flags
	
	def setupClContext(self):
		self.clPlatformList = cl.get_platforms()
		counter = 0
		for platform in self.clPlatformList:
			#~ tmp = self.clPlatformList[0]
			if self.clPlatform in platform.name.lower():
				self.platformIndex = counter
			counter = counter + 1
		clDevicesList = self.clPlatformList[self.platformIndex].get_devices()
		
		#~ vendorString = self.queue.device.vendor
		#~ # set work dimension of work group used in tracking kernel
		#~ if "intel" in vendorString.lower():  # work-around since the 'max_work_group_size' is not reported correctly for Intel-CPU using the AMD OpenCL driver (tested on: Intel(R) Core(TM) i5-3470)

		computeDeviceIdSelection = self.computeDeviceId # 0: AMD-GPU; 1: Intel CPU
		self.device = clDevicesList[computeDeviceIdSelection]
		# ipdb.set_trace()
		self.ctx = cl.Context([self.device])
		pass

	def plotCurrentMembraneCoordinates(self):
		host_membraneCoordinatesX = self.dev_membraneCoordinatesX.get()
		host_membraneCoordinatesY = self.dev_membraneCoordinatesY.get()
		plt.plot(host_membraneCoordinatesX,host_membraneCoordinatesY)

	def plotCurrentInterpolatedMembraneCoordinates(self):
		host_membranePolarCoordinates = self.dev_interpolatedMembraneCoordinates.get()
		plt.plot(host_membranePolarCoordinates['x'],host_membranePolarCoordinates['y'])

	#def 
	#	self.assertTrue(np.all(outputValue['x'] == referenceValue['x']))
	#	self.assertTrue(np.all(outputValue['y'] == referenceValue['y']))


if __name__ == '__main__':
	unittest.main()
