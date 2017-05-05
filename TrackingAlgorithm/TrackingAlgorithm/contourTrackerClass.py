import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
#~ import cv2 # OpenCV 2.3.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import PIL.Image as Image
import json
from scipy import signal
#~ import ConfigParser
import ipdb
import os
import time
from helpers import helpers
from mako.template import Template

class contourTracker( object ):
	def __init__(self, ctx, configReader, imageProcessor):
		self.configReader = configReader
		self.imageProcessor = imageProcessor
		self.setupClQueue(ctx)
		self.loadClKernels()
		self.loadConfig(configReader)
		self.setupTrackingParameters()
		self.setWorkGroupSizes()
		self.setupClTrackingVariables()
		self.setContourId(-1) # initialize the contour id to -1; this will later change at run time
		self.nrOfTrackingIterations = 0
		pass
	
	def setupClQueue(self,ctx):
		self.ctx = ctx
		self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
		self.mf = cl.mem_flags
		self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)
		pass
		
	def loadImage(self, imagePath):
		im = Image.open(imagePath)
		self.host_Img = self.imageProcessor.processImage(im)
		self.loadImageToGpu()
		pass
	
	def loadImageToGpu(self):
		self.dev_Img = cl.image_from_array(self.ctx, ary=self.host_Img, mode="r", norm_int=False, num_channels=1)
	
	def setContourId(self, id):
		self.id = id
	
	def getContourId(self):
		return self.id

	def loadClKernels(self):
		modulePath = __file__
		codePath, filename = os.path.split(modulePath) # get location of path where our tracking code is located
		clCodeFile = codePath+"/"+"clKernelCode.cl"
		fObj = open(clCodeFile, 'r')
		self.kernelString = "".join(fObj.readlines())
		self.applyTemplating()
		self.prg = cl.Program(self.ctx,self.kernelString).build()
		pass
	
	def applyTemplating(self):
		tpl = Template(self.kernelString)
		if self.configReader.positioningMethod == "maximumIntensityIncline":
			linear_fit_search_method="MAX_INCLINE_SEARCH"
		if self.configReader.positioningMethod == "meanIntensityIntercept":
			linear_fit_search_method="MIN_MAX_INTENSITY_SEARCH"
		rendered_tpl = tpl.render(linear_fit_search_method=linear_fit_search_method)
		self.kernelString=str(rendered_tpl)
		pass

	def loadConfig(self,configReader):
		self.startingCoordinate = configReader.startingCoordinate
		self.rotationCenterCoordinate = configReader.rotationCenterCoordinate
		
		self.linFitParameter = configReader.linFitParameter
		self.linFitSearchRange = configReader.linFitSearchRange
		self.interpolationFactor = configReader.interpolationFactor
		
		self.meanParameter = configReader.meanParameter
		self.meanRangePositionOffset = configReader.meanRangePositionOffset

		self.localAngleRange = configReader.localAngleRange
		self.nrOfLocalAngleSteps = configReader.nrOfLocalAngleSteps
		
		self.detectionKernelStrideSize = configReader.detectionKernelStrideSize
		self.nrOfStrides = configReader.nrOfStrides
		
		self.nrOfAnglesToCompare = configReader.nrOfAnglesToCompare
		
		self.nrOfIterationsPerContour = configReader.nrOfIterationsPerContour
		
		self.computeDeviceId = configReader.computeDeviceId

		self.inclineTolerance = configReader.inclineTolerance

		self.inclineRefinementRange = configReader.inclineRefinementRange
		
		self.coordinateTolerance = configReader.coordinateTolerance
		
		self.maxNrOfTrackingIterations = configReader.maxNrOfTrackingIterations
		self.minNrOfTrackingIterations = configReader.minNrOfTrackingIterations
		
		self.centerTolerance = configReader.centerTolerance
		
		self.maxInterCoordinateAngle = configReader.maxInterCoordinateAngle
		self.maxCoordinateShift = configReader.maxCoordinateShift
		
		self.resetNormalsAfterEachImage = configReader.resetNormalsAfterEachImage
		
	def setupTrackingParameters(self):
		### parameters for linear fit
		self.linFitParameter = self.linFitParameter*self.interpolationFactor/2
		self.linFitParameter = np.int32(self.linFitParameter)
		self.linFitSearchRange = self.linFitSearchRange/2
		self.linFitSearchRange = round(self.linFitSearchRange*self.interpolationFactor)
		self.linFitSearchRangeNrOfInterpolationPoints = 2*self.linFitSearchRange
		linFitSearchRangeStartXvalue =  -self.linFitSearchRange/self.interpolationFactor
		linFitSearchRangeEndXvalue =  self.linFitSearchRange/self.interpolationFactor
		
		self.linFitSearchRangeXvalues = np.float64(np.transpose(np.linspace(linFitSearchRangeStartXvalue,linFitSearchRangeEndXvalue,self.linFitSearchRangeNrOfInterpolationPoints)))
		self.buf_linFitSearchRangeXvalues = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.linFitSearchRangeXvalues)
		
		### parameters for calculating mean background intensity
		self.meanRangeXvalues = np.array(np.transpose(np.linspace(0,self.meanParameter,self.meanParameter)),dtype=np.float64)
		self.buf_meanRangeXvalues = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.meanRangeXvalues)

		### setup local rotation matrices
		startAngle = -self.localAngleRange/2
		endAngle = self.localAngleRange/2

		localAngles = np.linspace(startAngle,endAngle,self.nrOfLocalAngleSteps)

		self.localRotationMatrices = np.empty((localAngles.shape[0],2,2),np.dtype(np.float64))
		for index in range(localAngles.shape[0]):
			localAngle = localAngles[index]
			self.localRotationMatrices[index,:,:] = np.array([[np.cos(localAngle),-np.sin(localAngle)],[np.sin(localAngle),np.cos(localAngle)]])

		self.buf_localRotationMatrices = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.localRotationMatrices)

		### setup local rotation matrices
		initialNormalVector = self.startingCoordinate - self.rotationCenterCoordinate
		initialNormalVectorNorm = np.sqrt(initialNormalVector[0]**2 + initialNormalVector[1]**2)
		self.radiusUnitVector = initialNormalVector/initialNormalVectorNorm
		
		detectionStartAngle = 0
		detectionEndAngle = 2*np.pi

		self.nrOfDetectionAngleSteps = int(self.nrOfStrides*self.detectionKernelStrideSize)
		
		self.angleStepSize = np.float64((detectionEndAngle-detectionStartAngle)/self.nrOfDetectionAngleSteps)
		
		self.nrOfInterpolationPoints = np.int32(2*self.nrOfDetectionAngleSteps)
		
		self.imgSizeY = np.int32(self.linFitSearchRangeNrOfInterpolationPoints)
		self.imgSizeX = np.int32(self.nrOfLocalAngleSteps)
		
		pass

	def setupClTrackingVariables(self):
		self.host_fitIncline = np.empty(self.nrOfLocalAngleSteps,dtype=np.float64)
		self.dev_fitIncline = cl_array.to_device(self.queue, self.host_fitIncline)
		
		self.host_fitIntercept = np.empty(self.nrOfLocalAngleSteps,dtype=np.float64)
		self.dev_fitIntercept = cl_array.to_device(self.queue, self.host_fitIntercept)

		self.host_localMembranePositions = np.zeros(self.nrOfLocalAngleSteps, cl.array.vec.double2)
		self.dev_localMembranePositions = cl_array.to_device(self.queue, self.host_localMembranePositions)
		
		self.host_membraneCoordinatesX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneCoordinatesX[0] = self.startingCoordinate[0]
		self.dev_membraneCoordinatesX = cl_array.to_device(self.queue, self.host_membraneCoordinatesX)
		
		self.host_membraneCoordinatesY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneCoordinatesY[0] = self.startingCoordinate[1]
		self.dev_membraneCoordinatesY = cl_array.to_device(self.queue, self.host_membraneCoordinatesY)
		
		self.host_interCoordinateAngles = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_interCoordinateAngles = cl_array.to_device(self.queue, self.host_interCoordinateAngles)

		self.host_fitInclines = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_fitInclines = cl_array.to_device(self.queue, self.host_fitInclines)


		# these device arrays are not used on the host
		self.host_interpolatedMembraneCoordinatesX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_interpolatedMembraneCoordinatesX = cl_array.to_device(self.queue, self.host_interpolatedMembraneCoordinatesX)
		self.host_interpolatedMembraneCoordinatesY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_interpolatedMembraneCoordinatesY = cl_array.to_device(self.queue, self.host_interpolatedMembraneCoordinatesY)

		self.host_previousInterpolatedMembraneCoordinatesX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_previousInterpolatedMembraneCoordinatesX = cl_array.to_device(self.queue, self.host_previousInterpolatedMembraneCoordinatesX)
		self.host_previousInterpolatedMembraneCoordinatesY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_previousInterpolatedMembraneCoordinatesY = cl_array.to_device(self.queue, self.host_previousInterpolatedMembraneCoordinatesY)

		self.host_ds = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_ds = cl_array.to_device(self.queue, self.host_ds)

		self.host_sumds = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_sumds = cl_array.to_device(self.queue, self.host_sumds)
		
		self.host_membraneNormalVectorsX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneNormalVectorsX[0] = self.radiusUnitVector[0]
		self.dev_membraneNormalVectorsX = cl_array.to_device(self.queue, self.host_membraneNormalVectorsX)
		self.host_membraneNormalVectorsY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneNormalVectorsY[0] = self.radiusUnitVector[1]
		self.dev_membraneNormalVectorsY = cl_array.to_device(self.queue, self.host_membraneNormalVectorsY)
		
		self.host_closestLowerNoneNanIndex = np.int32(range(0,np.int32(self.nrOfDetectionAngleSteps)))
		self.dev_closestLowerNoneNanIndex = cl_array.to_device(self.queue, self.host_closestLowerNoneNanIndex)
		
		self.host_closestUpperNoneNanIndex = np.int32(range(0,np.int32(self.nrOfDetectionAngleSteps)))
		self.dev_closestUpperNoneNanIndex = cl_array.to_device(self.queue, self.host_closestUpperNoneNanIndex)
		
		self.host_membranePolarRadius = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarRadius = cl_array.to_device(self.queue, self.host_membranePolarRadius)
		self.dev_membranePolarRadiusTMP = cl_array.to_device(self.queue, self.host_membranePolarRadius)
		
		self.host_interpolatedMembranePolarRadius = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_interpolatedMembranePolarRadius = cl_array.to_device(self.queue, self.host_interpolatedMembranePolarRadius)

		self.host_membranePolarTheta = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarTheta = cl_array.to_device(self.queue, self.host_membranePolarTheta)
		self.dev_membranePolarThetaTMP = cl_array.to_device(self.queue, self.host_membranePolarTheta)
		
		self.host_angleDifferencesUpper = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_angleDifferencesUpper = cl_array.to_device(self.queue, self.host_angleDifferencesUpper)
		self.host_angleDifferencesLower = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_angleDifferencesLower = cl_array.to_device(self.queue, self.host_angleDifferencesLower)
				
		self.host_membranePolarRadiusInterpolation = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarRadiusInterpolation = cl_array.to_device(self.queue, self.host_membranePolarRadiusInterpolation)
		self.dev_membranePolarRadiusInterpolationTesting = cl_array.to_device(self.queue, self.host_membranePolarRadiusInterpolation)

		self.host_membranePolarThetaInterpolation = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarThetaInterpolation = cl_array.to_device(self.queue, self.host_membranePolarThetaInterpolation)
		self.dev_membranePolarThetaInterpolationTesting = cl_array.to_device(self.queue, self.host_membranePolarThetaInterpolation)

		startAngle = -np.pi
		endAngle = np.pi - 2*np.pi/self.nrOfDetectionAngleSteps # we substract, so that we have no angle overlap
		self.host_interpolationAngles = np.float64(np.linspace(startAngle,endAngle,self.nrOfDetectionAngleSteps))
		self.dev_interpolationAngles = cl_array.to_device(self.queue, self.host_interpolationAngles)
		
		self.host_dbgOut = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_dbgOut = cl_array.to_device(self.queue, self.host_dbgOut)
		
		self.host_dbgOut2 = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_dbgOut2 = cl_array.to_device(self.queue, self.host_dbgOut)
		
		########################################################################
		### setup OpenCL local memory
		########################################################################
		self.localMembranePositions_memSize = self.dev_localMembranePositions.nbytes * int(self.contourPointsPerWorkGroup)
		
		self.fitIncline_memSize = self.dev_fitIncline.nbytes * int(self.contourPointsPerWorkGroup)
		self.fitIntercept_memSize = self.dev_fitIntercept.nbytes * int(self.contourPointsPerWorkGroup)
		
		self.membranePolarTheta_memSize = self.dev_membranePolarTheta.nbytes
		self.membranePolarRadius_memSize = self.dev_membranePolarRadius.nbytes
		self.membranePolarThetaInterpolation_memSize = self.dev_membranePolarThetaInterpolation.nbytes
		self.membranePolarRadiusInterpolation_memSize = self.dev_membranePolarRadiusInterpolation.nbytes

		self.host_membraneNormalVectors = np.zeros(self.nrOfDetectionAngleSteps, cl.array.vec.double2)
		self.dev_membraneNormalVectors = cl_array.to_device(self.queue, self.host_membraneNormalVectors)
		self.membraneNormalVectors_memSize = self.dev_membraneNormalVectors.nbytes
		
		self.host_rotatedUnitVector = np.zeros(self.nrOfLocalAngleSteps, cl.array.vec.double2)
		self.dev_rotatedUnitVector = cl_array.to_device(self.queue, self.host_rotatedUnitVector)
		self.rotatedUnitVector_memSize = self.dev_rotatedUnitVector.nbytes * int(self.contourPointsPerWorkGroup)
		
		self.host_contourCenter = np.zeros(1, cl.array.vec.double2)
		self.dev_contourCenter = cl_array.to_device(self.queue, self.host_contourCenter)
		self.dev_previousContourCenter = cl_array.to_device(self.queue, self.host_contourCenter)
		
		self.host_listOfGoodCoordinates = np.zeros(self.nrOfDetectionAngleSteps, dtype=np.int32)
		self.dev_listOfGoodCoordinates = cl_array.to_device(self.queue, self.host_listOfGoodCoordinates)
		self.listOfGoodCoordinates_memSize = self.dev_listOfGoodCoordinates.nbytes
		
		### setup radial vectors used for linear interpolation
		self.host_radialVectors = np.zeros(shape=self.nrOfDetectionAngleSteps, dtype=cl.array.vec.double2)
		self.host_radialVectorsX = np.zeros(shape=self.nrOfDetectionAngleSteps, dtype=np.float64)
		self.host_radialVectorsY = np.zeros(shape=self.nrOfDetectionAngleSteps, dtype=np.float64)
		radiusUnitVector = np.array([1,0],dtype=np.float64)
		
		for counter, angle in enumerate(self.host_interpolationAngles):
			radiusVectorRotationMatrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
			rotatedRadiusUnitVector = radiusVectorRotationMatrix.dot(radiusUnitVector)
			self.host_radialVectors[counter] = rotatedRadiusUnitVector
			self.host_radialVectorsX[counter] = rotatedRadiusUnitVector[0]
			self.host_radialVectorsY[counter] = rotatedRadiusUnitVector[1]
		
		self.dev_radialVectors = cl_array.to_device(self.queue, self.host_radialVectors)
		self.dev_radialVectorsX = cl_array.to_device(self.queue, self.host_radialVectorsX)
		self.dev_radialVectorsY = cl_array.to_device(self.queue, self.host_radialVectorsY)
		pass

	def setWorkGroupSizes(self):
		self.global_size = (1,int(self.nrOfLocalAngleSteps))
		self.local_size = (1,int(self.nrOfLocalAngleSteps))
		self.gradientGlobalSize = (int(self.nrOfDetectionAngleSteps),1)
		
		vendorString = self.queue.device.vendor
		# set work dimension of work group used in tracking kernel
		if "intel" in vendorString.lower() or "nvidia" in vendorString.lower():  # work-around since the 'max_work_group_size' is not reported correctly for Intel-CPU using the AMD OpenCL driver (tested on: Intel(R) Core(TM) i5-3470)
			self.contourPointsPerWorkGroup = 256/self.nrOfLocalAngleSteps
		else:
			self.contourPointsPerWorkGroup = self.queue.device.max_work_group_size/self.nrOfLocalAngleSteps
		
		self.trackingWorkGroupSize = (int(self.contourPointsPerWorkGroup),int(self.nrOfLocalAngleSteps))
		self.trackingGlobalSize = (int(self.detectionKernelStrideSize),int(self.nrOfLocalAngleSteps))
		
	def setStartingCoordinates(self,dev_initialMembraneCoordinatesX,dev_initialMembraneCoordinatesY, \
									dev_initialMembranNormalVectorsX,dev_initialMembranNormalVectorsY):
		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesX.data,self.dev_membraneCoordinatesX.data).wait() #<-
		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesY.data,self.dev_membraneCoordinatesY.data).wait()
		cl.enqueue_copy_buffer(self.queue,dev_initialMembranNormalVectorsX.data,self.dev_membraneNormalVectorsX.data).wait()
		cl.enqueue_copy_buffer(self.queue,dev_initialMembranNormalVectorsY.data,self.dev_membraneNormalVectorsY.data).wait()
		barrierEvent = cl.enqueue_barrier(self.queue)
		self.queue.finish()
		
	def setStartingCoordinatesNew(self,dev_initialMembraneCoordinatesX,dev_initialMembraneCoordinatesY):
		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesX.data,self.dev_membraneCoordinatesX.data).wait() #<-
		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesY.data,self.dev_membraneCoordinatesY.data).wait()
		
		#cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesX.data,self.dev_interpolatedMembraneCoordinatesX.data).wait()
		#cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesY.data,self.dev_interpolatedMembraneCoordinatesY.data).wait()

		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesX.data,self.dev_previousInterpolatedMembraneCoordinatesX.data).wait()
		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesY.data,self.dev_previousInterpolatedMembraneCoordinatesY.data).wait()
		barrierEvent = cl.enqueue_barrier(self.queue)
		
	def setContourCenter(self,dev_initialContourCenter):
		cl.enqueue_copy_buffer(self.queue,dev_initialContourCenter.data,self.dev_previousContourCenter.data).wait()
		pass

	def setStartingMembraneNormals(self,dev_initialMembranNormalVectorsX,dev_initialMembranNormalVectorsY):
		if self.resetNormalsAfterEachImage and not self.getContourId()==0: # reset contour normal vector to radial vectors; we do this only starting for the second, since doing this for image 0, would destroy the correspondence of the indexes of the contour coordinates to their corresponding contour normals
			cl.enqueue_copy_buffer(self.queue,self.dev_radialVectorsX.data,self.dev_membraneNormalVectorsX.data).wait()
			cl.enqueue_copy_buffer(self.queue,self.dev_radialVectorsY.data,self.dev_membraneNormalVectorsY.data).wait()
		else: # copy contour normal vectors from last image to use as initial normal vectors for next image
			cl.enqueue_copy_buffer(self.queue,dev_initialMembranNormalVectorsX.data,self.dev_membraneNormalVectorsX.data).wait()
			cl.enqueue_copy_buffer(self.queue,dev_initialMembranNormalVectorsY.data,self.dev_membraneNormalVectorsY.data).wait()
		barrierEvent = cl.enqueue_barrier(self.queue)
		
	def getNrOfTrackingIterations(self):
		return self.nrOfTrackingIterations
		
	def resetNrOfTrackingIterations(self):
		self.nrOfTrackingIterations = 0

	def startTimer(self):
		self.startingTime = time.time()
		
	def getExectionTime(self):
		currentTime = time.time()
		return currentTime - self.startingTime
		
	def plotCurrentMembraneCoordinates(self):
		cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
		cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY)

	def plotCurrentInterpolatedMembraneCoordinates(self):
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY)

	def saveDeviceVariable(self,variableName,path):
		dev_variable = getattr(self,variableName)
		varOnHost = dev_variable.get()
		np.save(path+"/"+variableName+".npy", varOnHost, allow_pickle=True, fix_imports=True)
		pass

	def saveHostVariable(self,variableName,path):
		host_variable = getattr(self,variableName)
		np.save(path+"/"+variableName+".npy", host_variable, allow_pickle=True, fix_imports=True)
		pass
	
	def trackImage(self,imagePath):
		self.loadImage(imagePath)
		self.resetNrOfTrackingIterations()
		while(not self.checkTrackingFinished()): # start new tracking iteration with the previous contour as starting position
			self.trackContour()

	def trackContourSequentially(self):
		## tracking status variables
		#self.trackingFinished = np.int32(1) # True
		#self.iterationFinished = np.int32(1) # True

		for coordinateIndex in range(int(self.nrOfDetectionAngleSteps)):
			coordinateIndex = np.int32(coordinateIndex)
			
			angle = self.angleStepSize*np.float64(coordinateIndex+1)
			
			radiusVectorRotationMatrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
			
			self.dev_membraneNormalVectors = helpers.ToDoubleVectorOnDevice(self.queue,self.dev_membraneNormalVectorsX,self.dev_membraneNormalVectorsY)
			self.dev_membraneCoordinates = helpers.ToDoubleVectorOnDevice(self.queue,self.dev_membraneCoordinatesX,self.dev_membraneCoordinatesY)

			self.prg.findMembranePosition(self.queue, self.global_size, self.local_size, self.sampler, \
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
												coordinateIndex, \
												self.inclineTolerance, \
												self.inclineRefinementRange)

			barrierEvent = cl.enqueue_barrier(self.queue)

			self.dev_membraneCoordinatesX, self.dev_membraneCoordinatesY = helpers.ToSingleVectorsOnDevice(self.queue,self.dev_membraneCoordinates)
			self.dev_membraneNormalVectorsX, self.dev_membraneNormalVectorsY = helpers.ToSingleVectorsOnDevice(self.queue,self.dev_membraneNormalVectors)
			
			cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()

			cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()

			currentMembraneCoordinate = np.array([self.host_membraneCoordinatesX[coordinateIndex],self.host_membraneCoordinatesY[coordinateIndex]])
			
			radiusVector = currentMembraneCoordinate - self.rotationCenterCoordinate
			radiusVectorNorm = np.sqrt(radiusVector[0]**2 + radiusVector[1]**2)
			
			rotatedRadiusUnitVector = radiusVectorRotationMatrix.dot(self.radiusUnitVector)
			
			nextMembranePosition = self.rotationCenterCoordinate + rotatedRadiusUnitVector*radiusVectorNorm
			nextMembraneNormalVector = np.array([self.host_membraneNormalVectorsX[coordinateIndex],self.host_membraneNormalVectorsY[coordinateIndex]])
			
			if coordinateIndex < self.host_membraneCoordinatesX.shape[0]-1:
				self.host_membraneCoordinatesX[coordinateIndex+1] = nextMembranePosition[0]
				self.host_membraneCoordinatesY[coordinateIndex+1] = nextMembranePosition[1]
				
				self.host_membraneNormalVectorsX[coordinateIndex+1] = nextMembraneNormalVector[0]
				self.host_membraneNormalVectorsY[coordinateIndex+1] = nextMembraneNormalVector[1]

				self.dev_membraneCoordinatesX = cl_array.to_device(self.queue, self.host_membraneCoordinatesX)
				self.dev_membraneCoordinatesY = cl_array.to_device(self.queue, self.host_membraneCoordinatesY)
				
				self.dev_membraneNormalVectorsX = cl_array.to_device(self.queue, self.host_membraneNormalVectorsX)
				self.dev_membraneNormalVectorsY = cl_array.to_device(self.queue, self.host_membraneNormalVectorsY)
				
		# calculate new normal vectors
		self.dev_membraneCoordinates = helpers.ToDoubleVectorOnDevice(self.queue,self.dev_membraneCoordinatesX,self.dev_membraneCoordinatesY)
		self.dev_membraneNormalVectors = helpers.ToDoubleVectorOnDevice(self.queue,self.dev_membraneNormalVectorsX,self.dev_membraneNormalVectorsY)

		self.prg.calculateMembraneNormalVectors(self.queue, self.gradientGlobalSize, None, \
										   self.dev_membraneCoordinates.data, \
										   self.dev_membraneNormalVectors.data \
										  )

		self.calculateContourCenter()

		self.dev_membraneCoordinatesX, self.dev_membraneCoordinatesY = helpers.ToSingleVectorsOnDevice(self.queue,self.dev_membraneCoordinates)
		self.dev_membraneNormalVectorsX, self.dev_membraneNormalVectorsY = helpers.ToSingleVectorsOnDevice(self.queue,self.dev_membraneNormalVectors)

		cl.enqueue_copy_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.dev_interpolatedMembraneCoordinatesX.data).wait()
		cl.enqueue_copy_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.dev_interpolatedMembraneCoordinatesY.data).wait()

		cl.enqueue_copy_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.dev_previousInterpolatedMembraneCoordinatesX.data).wait()
		cl.enqueue_copy_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.dev_previousInterpolatedMembraneCoordinatesY.data).wait()
		
		self.setStartingCoordinatesNew(self.dev_interpolatedMembraneCoordinatesX, \
									   self.dev_interpolatedMembraneCoordinatesY)
		self.queue.finish()

	def trackContour(self):
		if self.resetNormalsAfterEachImage and not self.getContourId()==0 and self.nrOfTrackingIterations==0: # reset contour normal vector to radial vectors; we do this only starting for the second, since doing this for image 0, would destroy the correspondence of the indexes of the contour coordinates to their corresponding contour normals
			cl.enqueue_copy_buffer(self.queue,self.dev_radialVectorsX.data,self.dev_membraneNormalVectorsX.data).wait()
			cl.enqueue_copy_buffer(self.queue,self.dev_radialVectorsY.data,self.dev_membraneNormalVectorsY.data).wait()

		# tracking status variables
		self.nrOfTrackingIterations = self.nrOfTrackingIterations + 1
		
		stopInd = 1

		self.trackingFinished = np.array(1,dtype=np.int32) # True
		self.dev_trackingFinished = cl_array.to_device(self.queue, self.trackingFinished)
		
		self.iterationFinished = np.array(0,dtype=np.int32) # True
		self.dev_iterationFinished = cl_array.to_device(self.queue, self.iterationFinished)
		
		self.dev_membraneCoordinates = helpers.ToDoubleVectorOnDevice(self.queue,self.dev_membraneCoordinatesX,self.dev_membraneCoordinatesY)
		self.dev_membraneNormalVectors = helpers.ToDoubleVectorOnDevice(self.queue,self.dev_membraneNormalVectorsX,self.dev_membraneNormalVectorsY)
		self.dev_previousInterpolatedMembraneCoordinates = helpers.ToDoubleVectorOnDevice(self.queue,self.dev_previousInterpolatedMembraneCoordinatesX,self.dev_previousInterpolatedMembraneCoordinatesY)
		self.dev_membranePolarCoordinates = helpers.ToDoubleVectorOnDevice(self.queue,self.dev_membranePolarTheta,self.dev_membranePolarRadius)
		self.dev_interpolatedMembraneCoordinates = helpers.ToDoubleVectorOnDevice(self.queue,self.dev_interpolatedMembraneCoordinatesX,self.dev_interpolatedMembraneCoordinatesY)

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

		self.prg.filterNanValues(self.queue, self.gradientGlobalSize, None, \
								 self.dev_membraneCoordinates.data, \
								 self.dev_membraneNormalVectors.data, \
								 cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes) \
								 )
		barrierEvent = cl.enqueue_barrier(self.queue)

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

		self.prg.calculateInterCoordinateAngles(self.queue, self.gradientGlobalSize, None, \
												self.dev_interCoordinateAngles.data, \
												self.dev_membraneCoordinates.data \
											   )
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.prg.filterIncorrectCoordinates(self.queue, self.gradientGlobalSize, None, \
											self.dev_previousContourCenter.data, \
										    self.dev_interCoordinateAngles.data, \
										    self.dev_membraneCoordinates.data, \
										    self.dev_membraneNormalVectors.data, \
										    cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
										    self.maxInterCoordinateAngle \
										    )
		barrierEvent = cl.enqueue_barrier(self.queue)
		
		# information regarding barriers: http://stackoverflow.com/questions/13200276/what-is-the-difference-between-clenqueuebarrier-and-clfinish

		########################################################################
		### Calculate contour center
		########################################################################
		self.calculateContourCenter()

		########################################################################
		### Convert cartesian coordinates to polar coordinates
		########################################################################
		self.prg.cart2pol(self.queue, self.gradientGlobalSize, None, \
						  self.dev_membraneCoordinates.data, \
						  self.dev_membranePolarCoordinates.data, \
						  self.dev_contourCenter.data)
		barrierEvent = cl.enqueue_barrier(self.queue)

		########################################################################
		### Interpolate polar coordinates
		########################################################################
		self.prg.sortCoordinates(self.queue, (1,1), None, \
								self.dev_membranePolarCoordinates.data, \
								self.dev_membraneCoordinates.data, \
								self.dev_membraneNormalVectors.data, \
								np.int32(self.nrOfDetectionAngleSteps) \
								)
		barrierEvent = cl.enqueue_barrier(self.queue)
		
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

		########################################################################
		### Convert polar coordinates to cartesian coordinates
		########################################################################
		self.prg.checkIfTrackingFinished(self.queue, self.gradientGlobalSize, None, \
										 self.dev_interpolatedMembraneCoordinates.data, \
										 self.dev_previousInterpolatedMembraneCoordinates.data, \
										 self.dev_trackingFinished.data, \
										 self.coordinateTolerance)
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.prg.checkIfCenterConverged(self.queue, (1,1), None, \
										self.dev_contourCenter.data, \
										self.dev_previousContourCenter.data, \
										self.dev_trackingFinished.data, \
										self.centerTolerance)
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.dev_membraneNormalVectorsX, self.dev_membraneNormalVectorsY = helpers.ToSingleVectorsOnDevice(self.queue,self.dev_membraneNormalVectors)
		self.dev_previousInterpolatedMembraneCoordinatesX, self.dev_previousInterpolatedMembraneCoordinatesY = helpers.ToSingleVectorsOnDevice(self.queue,self.dev_previousInterpolatedMembraneCoordinates)
		self.dev_membraneCoordinatesX, self.dev_membraneCoordinatesY = helpers.ToSingleVectorsOnDevice(self.queue,self.dev_membraneCoordinates)
		self.dev_membranePolarTheta, self.dev_membranePolarRadius = helpers.ToSingleVectorsOnDevice(self.queue,self.dev_membranePolarCoordinates)
		self.dev_interpolatedMembraneCoordinatesX, self.dev_interpolatedMembraneCoordinatesY = helpers.ToSingleVectorsOnDevice(self.queue,self.dev_interpolatedMembraneCoordinates)

		cl.enqueue_read_buffer(self.queue, self.dev_trackingFinished.data, self.trackingFinished).wait()

		barrierEvent = cl.enqueue_barrier(self.queue)

		cl.enqueue_copy_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesX.data,self.dev_previousInterpolatedMembraneCoordinatesX.data).wait()
		cl.enqueue_copy_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesY.data,self.dev_previousInterpolatedMembraneCoordinatesY.data).wait()
		cl.enqueue_copy_buffer(self.queue,self.dev_contourCenter.data,self.dev_previousContourCenter.data).wait()

		# set variable to tell host program that the tracking iteration has finished
		basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/ReferenceDataForTests/UnitTests/OpenClKernels/setIterationFinished_000'
		path = basePath+'/input'

		self.saveDeviceVariable('dev_iterationFinished',path)

		self.prg.setIterationFinished(self.queue, (1,1), None, self.dev_iterationFinished.data)
		barrierEvent = cl.enqueue_barrier(self.queue)

		path = basePath+'/output'
		self.saveDeviceVariable('dev_iterationFinished',path)

		cl.enqueue_read_buffer(self.queue, self.dev_iterationFinished.data, self.iterationFinished).wait()

		self.setStartingCoordinatesNew(self.dev_interpolatedMembraneCoordinatesX, \
									   self.dev_interpolatedMembraneCoordinatesY)
		pass
		
	def calculateContourCenter(self):
		self.prg.calculateDs(self.queue, self.gradientGlobalSize, None, \
					   self.dev_membraneCoordinates.data, \
					   self.dev_ds.data \
					 )
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.prg.calculateSumDs(self.queue, self.gradientGlobalSize, None, \
					   self.dev_ds.data, self.dev_sumds.data \
					 )
		barrierEvent = cl.enqueue_barrier(self.queue)
		
		self.prg.calculateContourCenter(self.queue, (1,1), None, \
								   self.dev_membraneCoordinates.data, \
								   self.dev_ds.data, self.dev_sumds.data, \
								   self.dev_contourCenter.data, \
								   np.int32(self.nrOfDetectionAngleSteps) \
								  )
		barrierEvent = cl.enqueue_barrier(self.queue)

	def checkTrackingFinished(self):
		if self.nrOfTrackingIterations < self.minNrOfTrackingIterations:
			self.trackingFinished = 0 # force another iterations
		if self.nrOfTrackingIterations >= self.maxNrOfTrackingIterations:
			self.trackingFinished = 1 # force finish
		return self.trackingFinished
		pass
		
	def getMembraneCoordinatesX(self):
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		return self.host_interpolatedMembraneCoordinatesX/self.configReader.scalingFactor
		pass
	
	def getMembraneCoordinatesY(self):
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		return self.host_interpolatedMembraneCoordinatesY/self.configReader.scalingFactor
		pass

	def getMembraneCoordinatesXscaled(self):
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		return self.host_interpolatedMembraneCoordinatesX
		pass
	
	def getMembraneCoordinatesYscaled(self):
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		return self.host_interpolatedMembraneCoordinatesY
		pass

	def getMembraneNormalVectorsX(self):
		cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
		return self.host_membraneNormalVectorsX
		pass

	def getMembraneNormalVectorsY(self):
		cl.enqueue_read_buffer(self.queue,self.dev_membraneNormalVectorsY.data,self.host_membraneNormalVectorsY).wait()
		return self.host_membraneNormalVectorsY
		pass

	def getContourCenterCoordinates(self):
		cl.enqueue_read_buffer(self.queue, self.dev_contourCenter.data, self.host_contourCenter).wait()
		self.host_contourCenter[0]['x']=self.host_contourCenter[0]['x']/self.configReader.scalingFactor
		self.host_contourCenter[0]['y']=self.host_contourCenter[0]['y']/self.configReader.scalingFactor
		return self.host_contourCenter
		pass

	def getFitInclines(self):
		cl.enqueue_read_buffer(self.queue, self.dev_fitInclines.data, self.host_fitInclines).wait()
		return self.host_fitInclines * self.configReader.scalingFactor # needs to be multiplied, since putting in more pixels artificially reduces the value of the incline
		pass

