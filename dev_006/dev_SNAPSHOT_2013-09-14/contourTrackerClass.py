import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import cv2 # OpenCV 2.3.1
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Image
import json
#~ import ConfigParser
import ipdb
import os
import time

class contourTracker( object ):
	def __init__(self, ctx, config):
		self.setupClQueue(ctx)
		self.loadClKernels()
		self.loadConfig(config)
		self.setupTrackingParameters()
		self.setWorkGroupSizes()
		self.setupClTrackingVariables()
		self.loadBackground()
		self.setContourId(-1) # initialize the contour id to -1; this will later change at run time
		self.nrOfTrackingIterations = 0
		pass
	
	def setupClQueue(self,ctx):
		#~ self.ctx = cl.ctx([device])
		self.ctx = ctx
		self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
		self.mf = cl.mem_flags
		
		#~ self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.NONE,cl.filter_mode.LINEAR)
		#~ ipdb.set_trace()
		self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)
		pass
		
	def loadBackground(self):
		if self.backgroundImagePath == None:
			self.backgroundData = None
		else:
			bkgr = Image.open(self.backgroundImagePath)
			bkgrdata = list(bkgr.getdata()) 
			self.backgroundData = np.asarray(bkgrdata, dtype=np.float32).reshape(bkgr.size)
		
		#~ ipdb.set_trace()
		pass
		
	def loadImage(self, imagePath):
		im = Image.open(imagePath)
		imgdata = list(im.getdata()) 
		imageData = np.asarray(imgdata, dtype=np.float32).reshape(im.size)
		
		if self.backgroundData is None:
			self.host_Img = imageData
		else:
			self.host_Img = imageData/self.backgroundData
		
		self.dev_Img = cl.image_from_array(self.ctx, ary=self.host_Img, mode="r", norm_int=False, num_channels=1)
		
		#~ self.intensityRoiMatrix = self.host_Img[self.intensityRoiTopLeft[0]:self.intensityRoiBottomRight[0], self.intensityRoiTopLeft[1]:self.intensityRoiBottomRight[1]]
		#~ self.meanIntensity = np.float64(np.mean(self.intensityRoiMatrix))
		
		#~ ipdb.set_trace()
		#~ plt.imshow(self.intensityRoiMatrix)
		#~ plt.show()

		pass
	
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
		self.prg = cl.Program(self.ctx,self.kernelString).build()
		pass
	
	def loadConfig(self,config):
	# from: http://stackoverflow.com/questions/335695/lists-in-configparser
	# json.loads(self.config.get("SectionOne","startingCoordinate"))
		self.startingCoordinate = np.array(json.loads(config.get("TrackingParameters","startingCoordinate")))
		self.rotationCenterCoordinate = np.array(json.loads(config.get("TrackingParameters","rotationCenterCoordinate")))
		self.membraneNormalVector = np.array(json.loads(config.get("TrackingParameters","membraneNormalVector")))
		
		self.linFitParameter = json.loads(config.get("TrackingParameters","linFitParameter"))
		self.linFitSearchRange = json.loads(config.get("TrackingParameters","linFitSearchRange"))
		self.interpolationFactor = json.loads(config.get("TrackingParameters","interpolationFactor"))
		
		self.meanParameter = np.int32(json.loads(config.get("TrackingParameters","meanParameter")))
		self.meanRangePositionOffset = np.float64(json.loads(config.get("TrackingParameters","meanRangePositionOffset")))
		
		self.localAngleRange = np.float64(json.loads(config.get("TrackingParameters","localAngleRange")))
		self.nrOfLocalAngleSteps = np.int32(json.loads(config.get("TrackingParameters","nrOfLocalAngleSteps")))
		
		self.detectionKernelStrideSize = np.int32(json.loads(config.get("TrackingParameters","detectionKernelStrideSize")))
		self.nrOfStrides = np.int32(json.loads(config.get("TrackingParameters","nrOfStrides")))
		
		self.nrOfIterationsPerContour = np.int32(json.loads(config.get("TrackingParameters","nrOfIterationsPerContour")))
		
		backgroundImagePath = config.get("FileParameters","backgroundImagePath")
		if backgroundImagePath == "" or backgroundImagePath == "None":
			self.backgroundImagePath = None
		else:
			self.backgroundImagePath = json.loads(backgroundImagePath)
		
		self.computDeviceId = json.loads(config.get("OpenClParameters","computDeviceId"))
		
		self.coordinateTolerance = np.float64(json.loads(config.get("TrackingParameters","coordinateTolerance")))
		
		self.maxNrOfTrackingIterations = json.loads(config.get("TrackingParameters","maxNrOfTrackingIterations"))
		self.minNrOfTrackingIterations = json.loads(config.get("TrackingParameters","minNrOfTrackingIterations"))
		
		self.inclineTolerance = np.float64(json.loads(config.get("TrackingParameters","inclineTolerance")))
		
		#~ self.intensityRoiTopLeft = json.loads(config.get("TrackingParameters","intensityRoiTopLeft"))
		#~ self.intensityRoiBottomRight = json.loads(config.get("TrackingParameters","intensityRoiBottomRight"))
		
		self.centerTolerance = np.float64(json.loads(config.get("TrackingParameters","centerTolerance")))
		
		self.maxCoordinateAngle = np.float64(json.loads(config.get("TrackingParameters","maxCoordinateAngle")))
		
		#~ ipdb.set_trace()
		
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

		localRotationMatrices = np.empty((localAngles.shape[0],2,2),np.dtype(np.float64))
		for index in xrange(localAngles.shape[0]):
			localAngle = localAngles[index]
			localRotationMatrices[index,:,:] = np.array([[np.cos(localAngle),-np.sin(localAngle)],[np.sin(localAngle),np.cos(localAngle)]])

		#~ dev_localRotationMatrices = cl_array.to_device(ctx, queue, localRotationMatrices)
		self.buf_localRotationMatrices = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=localRotationMatrices)

		### setup local rotation matrices
		self.radiusUnitVector = self.membraneNormalVector
		
		detectionStartAngle = 0
		detectionEndAngle = 2*np.pi

		#~ nrOfDetectionAngleSteps = np.float64(8000);
		#~ nrOfDetectionAngleSteps = np.float64(2000);
		#~ nrOfDetectionAngleSteps = np.float64(1365); # this is roughly, where we currently run out of local memory; when we still used it in some kernels
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)
		
		self.angleStepSize = np.float64((detectionEndAngle-detectionStartAngle)/self.nrOfDetectionAngleSteps)
		
		self.nrOfInterpolationPoints = np.int32(2*self.nrOfDetectionAngleSteps)
		
		self.imgSizeY = np.int32(self.linFitSearchRangeNrOfInterpolationPoints)
		self.imgSizeX = np.int32(self.nrOfLocalAngleSteps)
		pass

	def setupClTrackingVariables(self):
		self.host_fitIncline = np.empty(self.nrOfLocalAngleSteps,dtype=np.float64)
		self.dev_fitIncline = cl_array.to_device(self.ctx, self.queue, self.host_fitIncline)
		
		self.host_fitIntercept = np.empty(self.nrOfLocalAngleSteps,dtype=np.float64)
		self.dev_fitIntercept = cl_array.to_device(self.ctx, self.queue, self.host_fitIntercept)
		
		self.host_localMembranePositionsX = np.zeros(shape=self.nrOfLocalAngleSteps,dtype=np.float64)
		#~ host_localMembranePositionsX = np.zeros(shape=self.localAngles.shape[0],dtype=np.float64)
		self.dev_localMembranePositionsX = cl_array.to_device(self.ctx, self.queue, self.host_localMembranePositionsX)

		self.host_localMembranePositionsY = np.zeros(shape=self.nrOfLocalAngleSteps,dtype=np.float64)
		#~ host_localMembranePositionsY = np.zeros(shape=localAngles.shape[0],dtype=np.float64)
		self.dev_localMembranePositionsY = cl_array.to_device(self.ctx, self.queue, self.host_localMembranePositionsY)

		self.host_membraneCoordinatesX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneCoordinatesX[0] = self.startingCoordinate[0]
		self.dev_membraneCoordinatesX = cl_array.to_device(self.ctx, self.queue, self.host_membraneCoordinatesX)
		
		self.host_membraneCoordinatesY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneCoordinatesY[0] = self.startingCoordinate[1]
		self.dev_membraneCoordinatesY = cl_array.to_device(self.ctx, self.queue, self.host_membraneCoordinatesY)
		
		self.host_interCoordinateAngles = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_interCoordinateAngles = cl_array.to_device(self.ctx, self.queue, self.host_interCoordinateAngles)
		

		# these device arrays are not used on the host
		self.host_interpolatedMembraneCoordinatesX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_interpolatedMembraneCoordinatesX = cl_array.to_device(self.ctx, self.queue, self.host_interpolatedMembraneCoordinatesX)
		self.host_interpolatedMembraneCoordinatesY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_interpolatedMembraneCoordinatesY = cl_array.to_device(self.ctx, self.queue, self.host_interpolatedMembraneCoordinatesY)

		self.host_previousInterpolatedMembraneCoordinatesX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_previousInterpolatedMembraneCoordinatesX = cl_array.to_device(self.ctx, self.queue, self.host_previousInterpolatedMembraneCoordinatesX)
		self.host_previousInterpolatedMembraneCoordinatesY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_previousInterpolatedMembraneCoordinatesY = cl_array.to_device(self.ctx, self.queue, self.host_previousInterpolatedMembraneCoordinatesY)

		self.host_ds = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_ds = cl_array.to_device(self.ctx, self.queue, self.host_ds)

		self.host_sumds = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_sumds = cl_array.to_device(self.ctx, self.queue, self.host_sumds)
		
		self.host_membraneNormalVectorsX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneNormalVectorsX[0] = self.radiusUnitVector[0]
		self.dev_membraneNormalVectorsX = cl_array.to_device(self.ctx, self.queue, self.host_membraneNormalVectorsX)
		self.host_membraneNormalVectorsY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneNormalVectorsY[0] = self.radiusUnitVector[1]
		self.dev_membraneNormalVectorsY = cl_array.to_device(self.ctx, self.queue, self.host_membraneNormalVectorsY)
		
		self.host_closestLowerNoneNanIndex = np.int32(range(0,np.float32(self.nrOfDetectionAngleSteps)))
		self.dev_closestLowerNoneNanIndex = cl_array.to_device(self.ctx, self.queue, self.host_closestLowerNoneNanIndex)
		
		self.host_closestUpperNoneNanIndex = np.int32(range(0,np.float32(self.nrOfDetectionAngleSteps)))
		self.dev_closestUpperNoneNanIndex = cl_array.to_device(self.ctx, self.queue, self.host_closestUpperNoneNanIndex)
		
		self.host_membranePolarRadius = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarRadius = cl_array.to_device(self.ctx, self.queue, self.host_membranePolarRadius)
		self.dev_membranePolarRadiusTMP = cl_array.to_device(self.ctx, self.queue, self.host_membranePolarRadius)
		
		self.host_interpolatedMembranePolarRadius = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_interpolatedMembranePolarRadius = cl_array.to_device(self.ctx, self.queue, self.host_interpolatedMembranePolarRadius)

		self.host_membranePolarTheta = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarTheta = cl_array.to_device(self.ctx, self.queue, self.host_membranePolarTheta)
		self.dev_membranePolarThetaTMP = cl_array.to_device(self.ctx, self.queue, self.host_membranePolarTheta)
		
		self.host_membranePolarRadiusInterpolation = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarRadiusInterpolation = cl_array.to_device(self.ctx, self.queue, self.host_membranePolarRadiusInterpolation)
		self.dev_membranePolarRadiusInterpolationTesting = cl_array.to_device(self.ctx, self.queue, self.host_membranePolarRadiusInterpolation)

		self.host_membranePolarThetaInterpolation = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarThetaInterpolation = cl_array.to_device(self.ctx, self.queue, self.host_membranePolarThetaInterpolation)
		self.dev_membranePolarThetaInterpolationTesting = cl_array.to_device(self.ctx, self.queue, self.host_membranePolarThetaInterpolation)

		#~ host_interpolationAngles = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		#~ host_interpolationAngles = angleStepSize*np.float64(range(self.nrOfDetectionAngleSteps))
		startAngle = -np.pi
		endAngle = np.pi - 2*np.pi/self.nrOfDetectionAngleSteps # we substract, so that we have no angle overlap
		#~ self.host_interpolationAngles = np.float64(np.linspace(-np.pi,np.pi,self.nrOfDetectionAngleSteps))
		self.host_interpolationAngles = np.float64(np.linspace(startAngle,endAngle,self.nrOfDetectionAngleSteps))
		self.dev_interpolationAngles = cl_array.to_device(self.ctx, self.queue, self.host_interpolationAngles)

		self.host_b = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_b = cl_array.to_device(self.ctx, self.queue, self.host_b)

		self.host_c = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_c = cl_array.to_device(self.ctx, self.queue, self.host_c)

		self.host_d = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_d = cl_array.to_device(self.ctx, self.queue, self.host_d)

		self.host_dbgOut = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_dbgOut = cl_array.to_device(self.ctx, self.queue, self.host_dbgOut)
		
		self.host_dbgOut2 = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_dbgOut2 = cl_array.to_device(self.ctx, self.queue, self.host_dbgOut)
		
		########################################################################
		### setup OpenCL local memory
		########################################################################
		#~ self.contourPointsPerWorkGroup
		self.localMembranePositions_memSize = self.dev_localMembranePositionsX.nbytes * int(self.contourPointsPerWorkGroup)
		self.fitIncline_memSize = self.dev_fitIncline.nbytes * int(self.contourPointsPerWorkGroup)
		self.fitIntercept_memSize = self.dev_fitIntercept.nbytes * int(self.contourPointsPerWorkGroup)
		self.fitIntercept_memSize = self.dev_fitIntercept.nbytes * int(self.contourPointsPerWorkGroup)
		
		self.membranePolarTheta_memSize = self.dev_membranePolarTheta.nbytes
		self.membranePolarRadius_memSize = self.dev_membranePolarRadius.nbytes
		#~ self.membranePolarThetaInterpolation_memSize = self.dev_membranePolarThetaInterpolation.nbytes
		#~ self.membranePolarRadiusInterpolation_memSize = self.dev_membranePolarRadiusInterpolation.nbytes
		
		self.host_membraneNormalVectorsX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneNormalVectors = np.zeros(self.nrOfDetectionAngleSteps, cl.array.vec.double2)
		self.dev_membraneNormalVectors = cl_array.to_device(self.ctx, self.queue, self.host_membraneNormalVectors)
		self.membraneNormalVectors_memSize = self.dev_membraneNormalVectors.nbytes


		#~ self.host_fitIncline = np.empty(self.nrOfLocalAngleSteps,dtype=np.float64)
		self.host_rotatedUnitVector = np.zeros(self.nrOfLocalAngleSteps, cl.array.vec.double2)
		self.dev_rotatedUnitVector = cl_array.to_device(self.ctx, self.queue, self.host_rotatedUnitVector)
		self.rotatedUnitVector_memSize = self.dev_rotatedUnitVector.nbytes * int(self.contourPointsPerWorkGroup)
		
		self.host_contourCenter = np.zeros(1, cl.array.vec.double2)
		self.dev_contourCenter = cl_array.to_device(self.ctx, self.queue, self.host_contourCenter)
		self.dev_previousContourCenter = cl_array.to_device(self.ctx, self.queue, self.host_contourCenter)
		
		pass

	def setWorkGroupSizes(self):
		self.global_size = (1,int(self.nrOfLocalAngleSteps))
		self.local_size = (1,int(self.nrOfLocalAngleSteps))
		self.gradientGlobalSize = (int(self.nrOfDetectionAngleSteps),1)
		
		#~ ipdb.set_trace()
		# set work dimension of work group used in tracking kernel
		self.contourPointsPerWorkGroup = self.queue.device.max_work_group_size/self.nrOfLocalAngleSteps
		self.trackingWorkGroupSize = (int(self.contourPointsPerWorkGroup),int(self.nrOfLocalAngleSteps))
		self.trackingGlobalSize = (int(self.detectionKernelStrideSize),int(self.nrOfLocalAngleSteps))
		
	def setStartingCoordinates(self,dev_initialMembraneCoordinatesX,dev_initialMembraneCoordinatesY, \
									dev_initialMembranNormalVectorsX,dev_initialMembranNormalVectorsY):
		#~ self.dev_membraneCoordinatesX = dev_initialMembraneCoordinatesX
		#~ self.dev_membraneCoordinatesY = dev_initialMembraneCoordinatesY
		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesX.data,self.dev_membraneCoordinatesX.data).wait() #<-
		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesY.data,self.dev_membraneCoordinatesY.data).wait()
		#~ self.dev_membraneNormalVectorsX = dev_initialMembranNormalVectorsX
		#~ self.dev_membraneNormalVectorsY = dev_initialMembranNormalVectorsY
		#~ ipdb.set_trace()
		cl.enqueue_copy_buffer(self.queue,dev_initialMembranNormalVectorsX.data,self.dev_membraneNormalVectorsX.data).wait()
		cl.enqueue_copy_buffer(self.queue,dev_initialMembranNormalVectorsY.data,self.dev_membraneNormalVectorsY.data).wait()
		self.queue.finish()
		
	def setStartingCoordinatesNew(self,dev_initialMembraneCoordinatesX,dev_initialMembraneCoordinatesY):
		#~ self.dev_membraneCoordinatesX = dev_initialMembraneCoordinatesX
		#~ self.dev_membraneCoordinatesY = dev_initialMembraneCoordinatesY
		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesX.data,self.dev_membraneCoordinatesX.data).wait() #<-
		cl.enqueue_copy_buffer(self.queue,dev_initialMembraneCoordinatesY.data,self.dev_membraneCoordinatesY.data).wait()
		#~ self.dev_membraneNormalVectorsX = dev_initialMembranNormalVectorsX
		#~ self.dev_membraneNormalVectorsY = dev_initialMembranNormalVectorsY
		#~ ipdb.set_trace()
		#~ self.queue.finish()
		
	def setStartingMembraneNormals(self,dev_initialMembranNormalVectorsX,dev_initialMembranNormalVectorsY):
		cl.enqueue_copy_buffer(self.queue,dev_initialMembranNormalVectorsX.data,self.dev_membraneNormalVectorsX.data).wait()
		cl.enqueue_copy_buffer(self.queue,dev_initialMembranNormalVectorsY.data,self.dev_membraneNormalVectorsY.data).wait()
		#~ self.queue.finish()
		
	def getNrOfTrackingIterations(self):
		return self.nrOfTrackingIterations
		
	def resetNrOfTrackingIterations(self):
		self.nrOfTrackingIterations = 0

	def startTimer(self):
		self.startingTime = time.time()
		
	def getExectionTime(self):
		currentTime = time.time()
		return currentTime - self.startingTime
		
	def trackContour(self):
		# tracking status variables
		self.nrOfTrackingIterations = self.nrOfTrackingIterations + 1
		
		stopInd = 1
		#~ if self.nrOfTrackingIterations>=stopInd and self.getContourId() is 248:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_ds.data, self.host_ds).wait()
			#~ ipdb.set_trace()
			
		#~ self.host_trackingFinished = np.int32(0) # True
		self.trackingFinished = np.array(1,dtype=np.int32) # True
		self.dev_trackingFinished = cl_array.to_device(self.ctx, self.queue, self.trackingFinished)
		
		#~ self.iterationFinished = np.int32(0) # True
		self.iterationFinished = np.array(0,dtype=np.int32) # True
		self.dev_iterationFinished = cl_array.to_device(self.ctx, self.queue, self.iterationFinished)
		
		#~ print "self.iterationFinished:"
		#~ print self.iterationFinished
		
		#~ ipdb.set_trace()
		
		#~ for void in xrange(0,nrOfIterationsPerContour):
		#for coordinateIndex in xrange(self.nrOfDetectionAngleSteps):
			
			#coordinateIndex = np.int32(coordinateIndex)
			
			#self.prg.findMembranePosition(self.queue, self.global_size, self.local_size, self.sampler, \
								 #self.dev_Img, self.imgSizeX, self.imgSizeY, \
								 #self.buf_localRotationMatrices, \
								 #self.buf_linFitSearchRangeXvalues, \
								 #self.linFitParameter, \
								 ##~ cl.LocalMemory(fitIntercept_memSize), cl.LocalMemory(fitIncline_memSize), \
								 #self.dev_fitIntercept.data, self.dev_fitIncline.data, \
								 #self.meanParameter, \
								 #self.buf_meanRangeXvalues, self.meanRangePositionOffset, \
								 ##~ cl.LocalMemory(localMembranePositions_memSize), cl.LocalMemory(localMembranePositions_memSize), \
								 #self.dev_localMembranePositionsX.data, self.dev_localMembranePositionsY.data, \
								 #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								 #self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
								 #coordinateIndex)
		
		#barrierEvent = cl.enqueue_barrier(self.queue)

		#self.host_membraneCoordinatesXdebug = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		#self.host_membraneCoordinatesYdebug = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesXdebug).wait()
		#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesYdebug).wait()

		##~ self.dev_membraneCoordinatesX = cl_array.to_device(self.ctx, self.queue, self.host_membraneCoordinatesX)
		
		#self.queue.finish()
		
		#~ print "xCoord[0]:"+str(self.host_membraneCoordinatesXdebug[0])
		#~ print "yCoord[0]:"+str(self.host_membraneCoordinatesYdebug[0])
		
		for strideNr in xrange(self.nrOfStrides):
			
			# set the starting index of the coordinate array for each kernel instance
			kernelCoordinateStartingIndex = np.int32(strideNr*self.detectionKernelStrideSize)
			#~ print kernelCoordinateStartingIndex
			
			#self.prg.findMembranePositionNew(self.queue, self.trackingGlobalSize, self.trackingWorkGroupSize, self.sampler, \
											 #self.dev_Img, self.imgSizeX, self.imgSizeY, \
											 #self.buf_localRotationMatrices, \
											 #self.buf_linFitSearchRangeXvalues, \
											 #self.linFitParameter, \
											 #cl.LocalMemory(self.fitIntercept_memSize), cl.LocalMemory(self.fitIncline_memSize), \
											 ##~ self.dev_fitIntercept.data, self.dev_fitIncline.data, \
											 #self.meanParameter, \
											 #self.buf_meanRangeXvalues, self.meanRangePositionOffset, \
											 #cl.LocalMemory(self.localMembranePositions_memSize), cl.LocalMemory(self.localMembranePositions_memSize), \
											 ##~ self.dev_localMembranePositionsX.data, self.dev_localMembranePositionsY.data, \
											 #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
											 #self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
											 #kernelCoordinateStartingIndex)

			self.prg.findMembranePositionNew2(self.queue, self.trackingGlobalSize, self.trackingWorkGroupSize, self.sampler, \
											 self.dev_Img, self.imgSizeX, self.imgSizeY, \
											 self.buf_localRotationMatrices, \
											 self.buf_linFitSearchRangeXvalues, \
											 self.linFitParameter, \
											 cl.LocalMemory(self.fitIntercept_memSize), cl.LocalMemory(self.fitIncline_memSize), \
											 cl.LocalMemory(self.rotatedUnitVector_memSize), \
											 #~ self.dev_fitIntercept.data, self.dev_fitIncline.data, \
											 self.meanParameter, \
											 self.buf_meanRangeXvalues, self.meanRangePositionOffset, \
											 cl.LocalMemory(self.localMembranePositions_memSize), cl.LocalMemory(self.localMembranePositions_memSize), \
											 #~ self.dev_localMembranePositionsX.data, self.dev_localMembranePositionsY.data, \
											 self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
											 self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
											 kernelCoordinateStartingIndex, \
											 self.inclineTolerance)
											 
			#self.prg.findMembranePositionNew3(self.queue, self.trackingGlobalSize, self.trackingWorkGroupSize, self.sampler, \
											 #self.dev_Img, self.imgSizeX, self.imgSizeY, \
											 #self.buf_localRotationMatrices, \
											 #self.buf_linFitSearchRangeXvalues, \
											 #self.linFitParameter, \
											 #cl.LocalMemory(self.fitIntercept_memSize), cl.LocalMemory(self.fitIncline_memSize), \
											 #cl.LocalMemory(self.rotatedUnitVector_memSize), \
											 ##~ self.dev_fitIntercept.data, self.dev_fitIncline.data, \
											 #self.meanParameter, \
											 #self.buf_meanRangeXvalues, self.meanRangePositionOffset, \
											 #cl.LocalMemory(self.localMembranePositions_memSize), cl.LocalMemory(self.localMembranePositions_memSize), \
											 ##~ self.dev_localMembranePositionsX.data, self.dev_localMembranePositionsY.data, \
											 #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
											 #self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
											 #kernelCoordinateStartingIndex, \
											 #self.inclineTolerance, \
											 #self.meanIntensity \
											 #)


			#~ if self.nrOfTrackingIterations>=stopInd and self.getContourId() is 248:
				#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
				#~ cl.enqueue_read_buffer(self.queue, self.dev_ds.data, self.host_ds).wait()
				#~ ipdb.set_trace()

		barrierEvent = cl.enqueue_barrier(self.queue)
		
		#self.prg.filterNanValues(self.queue, self.gradientGlobalSize, None, \
								 ##~ cl.LocalMemory(self.localMembranePositions_memSize), cl.LocalMemory(self.localMembranePositions_memSize), \
								 #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								 #self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
								 #self.dev_closestLowerNoneNanIndex.data, self.dev_closestUpperNoneNanIndex.data \
								 #)
		self.prg.filterNanValues(self.queue, self.gradientGlobalSize, None, \
								 self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								 self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
								 cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes) \
								 #~ self.dev_dbgOut.data, \
								 #~ self.dev_dbgOut2.data \
								 )
		
		#~ if self.nrOfTrackingIterations>=stopInd and self.getContourId() is 248:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_ds.data, self.host_ds).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut.data, self.host_dbgOut).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut2.data, self.host_dbgOut2).wait()
			#~ ipdb.set_trace()


			#~ barrierEvent = cl.enqueue_barrier(self.queue)
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
		#~ 
			#~ ipdb.set_trace()
		
		barrierEvent = cl.enqueue_barrier(self.queue)
		
		
		#~ ipdb.set_trace()
		self.prg.calculateInterCoordinateAngles(self.queue, self.gradientGlobalSize, None, \
												self.dev_interCoordinateAngles.data, \
												self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data \
											   )

		barrierEvent = cl.enqueue_barrier(self.queue)

		#~ cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interCoordinateAngles).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
#~ 
		#~ ipdb.set_trace()
		
		self.prg.filterIncorrectCoordinates(self.queue, self.gradientGlobalSize, None, \
						 self.dev_interCoordinateAngles.data, \
						 self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
						 self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
						 cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
						 self.maxCoordinateAngle \
						 #~ self.dev_dbgOut.data, \
						 #~ self.dev_dbgOut2.data \
						 )

		barrierEvent = cl.enqueue_barrier(self.queue)
		
		#~ cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interCoordinateAngles).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
#~ 
		#~ ipdb.set_trace()
		#~ self.prg.calculateInterCoordinateAngles(self.queue, self.gradientGlobalSize, None, \
												#~ self.dev_interCoordinateAngles.data, \
												#~ self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data \
											   #~ )
		#~ cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interCoordinateAngles).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
		#~ ipdb.set_trace()
		
		#~ plt.plot(self.host_interCoordinateAngles), plt.show()
		#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()


		#~ self.queue.finish()

		#~ ipdb.set_trace()
		
		#~ plt.imshow(self.host_Img)
		#~ ax = plt.gca()
		#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,'k')
		#~ plt.show()
		
		# information regarding barriers: http://stackoverflow.com/questions/13200276/what-is-the-difference-between-clenqueuebarrier-and-clfinish

		## calculate new normal vectors
		#self.prg.calculateMembraneNormalVectors(self.queue, self.gradientGlobalSize, None, \
										   #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
										   #self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data \
										   ##~ cl.LocalMemory(membraneNormalVectors_memSize) \
										  #)
										  
		#barrierEvent = cl.enqueue_barrier(self.queue)
		
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()

		#~ ipdb.set_trace()

		#~ plt.imshow(self.host_Img)
		#~ ax = plt.gca()
		#~ ax.invert_yaxis() # needed so that usage of 'plt.quiver' (below), will play along
		#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,'k')
		#~ plt.show()

	########################################################################
	### Calculate contour center
	########################################################################
		
		### Use this for CPU and when number of detected points <1024
		#if self.computDeviceId is 10:
			##~ print bla
			#self.prg.calculateContourCenter(self.queue, self.gradientGlobalSize, self.gradientGlobalSize, \
										   #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
										   #cl.LocalMemory(self.membraneNormalVectors_memSize), cl.LocalMemory(self.membraneNormalVectors_memSize), \
										   #self.dev_contourCenter.data \
										  #)
		
		#### Use this for GPU and when number of detected points >500
		#### NOTE: There is a in the OpenCL driver for the Intel CPU. So that in the funciton below,
		#### 	  the CLK_GLOBAL_MEM_FENCE is not respected correctly leading to incorrect results
		#if self.computDeviceId is 10:
			#self.prg.calculateContourCenterNew(self.queue, self.gradientGlobalSize, None, \
											   #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
											   #self.dev_ds.data, self.dev_sumds.data, \
											   #self.dev_contourCenter.data \
											  #)
		
		#barrierEvent = cl.enqueue_barrier(self.queue)
		
		self.prg.calculateDs(self.queue, self.gradientGlobalSize, None, \
					   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
					   self.dev_ds.data \
					 )

		#~ if self.nrOfTrackingIterations>=stopInd:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_ds.data, self.host_ds).wait()
			#~ ipdb.set_trace()


		#~ barrierEvent = cl.enqueue_barrier(self.queue)

		self.prg.calculateSumDs(self.queue, self.gradientGlobalSize, None, \
					   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
					   self.dev_ds.data, self.dev_sumds.data \
					 )

		#~ barrierEvent = cl.enqueue_barrier(self.queue)
		
		self.prg.calculateContourCenterNew2(self.queue, (1,1), None, \
								   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								   self.dev_ds.data, self.dev_sumds.data, \
								   self.dev_contourCenter.data, \
								   np.int32(self.nrOfDetectionAngleSteps) \
								  )
		
		#~ if self.nrOfTrackingIterations>=stopInd:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
			#~ ipdb.set_trace()

		
		#~ self.queue.finish()
		#~ ipdb.set_trace()
		########################################################################
		### Convert cartesian coordinates to polar coordinates
		########################################################################
		self.prg.cart2pol(self.queue, self.gradientGlobalSize, None, \
						  self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
						  self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
						  self.dev_contourCenter.data)

		#~ outOfOrderProfilingQueue.finish()
		#~ ipdb.set_trace()
		#~ dev_contourCenter
		
		barrierEvent = cl.enqueue_barrier(self.queue)

		#~ if self.nrOfTrackingIterations>=stopInd:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
			#~ ipdb.set_trace()

	########################################################################
	### Interpolate polar coordinates
	########################################################################
		#~ cl.enqueue_read_buffer(outOfOrderProfilingQueue, dev_membranePolarRadius.data, host_membranePolarRadius).wait()
		#~ cl.enqueue_read_buffer(outOfOrderProfilingQueue, dev_membranePolarTheta.data, host_membranePolarTheta).wait()
		#~ outOfOrderProfilingQueue.finish()
		#~ ipdb.set_trace()
		#~ plt.plot(host_membranePolarTheta,host_membranePolarRadius), plt.show()

		#~ cl.LocalMemory(membranePolarThetaInterpolation_memSize)
		#~ cl.LocalMemory(membranePolarRadiusInterpolation_memSize)
		
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
		#~ ipdb.set_trace()
		#~ plt.plot(self.host_membranePolarTheta,self.host_membranePolarRadius,'r')
		#~ plt.show()

		#~ self.prg.sortPolarCoordinates(self.queue, (1,1), None, \
									  #~ self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
									  #~ np.int32(self.nrOfDetectionAngleSteps) \
									 #~ )
		
		#~ self.prg.sortPolarCoordinatesNew(self.queue, self.gradientGlobalSize, (256,1), \
									  #~ self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
									  #~ cl.LocalMemory(self.membranePolarTheta_memSize), cl.LocalMemory(self.membranePolarRadius_memSize), \
									  #~ np.int32(self.nrOfDetectionAngleSteps) \
									 #~ )

		self.prg.sortCoordinates(self.queue, (1,1), None, \
								self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
								self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
								np.int32(self.nrOfDetectionAngleSteps) \
								)
		
		#~ ipdb.set_trace()
		
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
		#~ ipdb.set_trace()
		#~ plt.plot(self.host_membranePolarTheta,self.host_membranePolarRadius,'r')
		#~ plt.show()
		#~ 
		barrierEvent = cl.enqueue_barrier(self.queue)
		#~ ipdb.set_trace()
		
		#~ if self.nrOfTrackingIterations>=stopInd:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
			#~ ipdb.set_trace()

		
		self.prg.interpolatePolarCoordinates(self.queue, self.gradientGlobalSize, None, \
											self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
											#~ cl.LocalMemory(dev_membranePolarRadius.nbytes), \
											#~ cl.LocalMemory(dev_membranePolarTheta.nbytes), \
											self.dev_membranePolarRadiusTMP.data, \
											self.dev_membranePolarThetaTMP.data, \
											#~ cl.LocalMemory(membranePolarRadiusInterpolation_memSize), \
											#~ cl.LocalMemory(membranePolarThetaInterpolation_memSize), \
											self.dev_membranePolarRadiusInterpolation.data, \
											self.dev_membranePolarThetaInterpolation.data, \
											self.dev_membranePolarRadiusInterpolationTesting.data, \
											self.dev_membranePolarThetaInterpolationTesting.data, \
											self.dev_interpolationAngles.data, \
											self.nrOfInterpolationPoints, \
											np.int32(self.nrOfDetectionAngleSteps), \
											#~ self.dev_dbgOut.data, \
											self.dev_interpolatedMembranePolarRadius.data, \
											self.dev_b.data, self.dev_c.data, self.dev_d.data \
											)
		
		#~ ipdb.set_trace()
		
		barrierEvent = cl.enqueue_barrier(self.queue)
		
		#~ if self.nrOfTrackingIterations>=stopInd:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
			#~ ipdb.set_trace()
		
		########################################################################
		### Convert polar coordinates to cartesian coordinates
		########################################################################
		#~ self.prg.pol2cart(self.queue, self.gradientGlobalSize, None, \
					 #~ self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
					 #~ self.dev_membranePolarRadius.data,	self.dev_membranePolarTheta.data, \
					 #~ self.dev_contourCenter.data  )
		#~ self.prg.pol2cart(self.queue, self.gradientGlobalSize, None, \
					 #~ self.dev_interpolatedMembraneCoordinatesX.data, self.dev_interpolatedMembraneCoordinatesY.data, \
					 #~ self.dev_interpolatedMembranePolarRadius.data,	self.dev_membranePolarTheta.data, \
					 #~ self.dev_contourCenter.data  )
		self.prg.pol2cart(self.queue, self.gradientGlobalSize, None, \
						  self.dev_interpolatedMembraneCoordinatesX.data, self.dev_interpolatedMembraneCoordinatesY.data, \
						  self.dev_interpolatedMembranePolarRadius.data, self.dev_interpolationAngles.data, \
						  self.dev_contourCenter.data  )

		barrierEvent = cl.enqueue_barrier(self.queue)
		
		#~ if self.nrOfTrackingIterations>=stopInd:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
			#~ ipdb.set_trace()

		
		# calculate new normal vectors
		#if self.nrOfTrackingIterations<10: # ToDo: If final, add parameter to for when to turn of the calculation of the membrane normals
			##~ ipdb.set_trace()
			#self.prg.calculateMembraneNormalVectors(self.queue, self.gradientGlobalSize, None, \
												    #self.dev_interpolatedMembraneCoordinatesX.data, self.dev_interpolatedMembraneCoordinatesY.data, \
												    #self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data \
												    ##~ cl.LocalMemory(membraneNormalVectors_memSize) \
												   #)
										  
		barrierEvent = cl.enqueue_barrier(self.queue)

		
		#~ barrierEvent = cl.enqueue_barrier(self.queue)
		
		self.prg.checkIfTrackingFinished(self.queue, self.gradientGlobalSize, None, \
										 self.dev_interpolatedMembraneCoordinatesX.data, \
										 self.dev_interpolatedMembraneCoordinatesY.data, \
										 self.dev_previousInterpolatedMembraneCoordinatesX.data, \
										 self.dev_previousInterpolatedMembraneCoordinatesY.data, \
										 self.dev_trackingFinished.data, \
										 self.coordinateTolerance)

		self.prg.checkIfCenterConverged(self.queue, (1,1), None, \
										self.dev_contourCenter.data, \
										self.dev_previousContourCenter.data, \
										self.dev_trackingFinished.data, \
										self.centerTolerance)
		
		cl.enqueue_read_buffer(self.queue, self.dev_trackingFinished.data, self.trackingFinished).wait()
		
		#~ ipdb.set_trace()
		
		#~ barrierEvent = cl.enqueue_barrier(self.queue)
		#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut.data, self.host_dbgOut).wait()
		
		#~ self.queue.finish()
		#~ print "self.iterationFinished:"
		#~ print self.iterationFinished
		
		# write result of the interpolated contour to the arrays used checking if tracking finished
		#~ self.dev_previousInterpolatedMembraneCoordinatesX.data = self.dev_interpolatedMembraneCoordinatesX.data
		#~ self.dev_previousInterpolatedMembraneCoordinatesY.data = self.dev_interpolatedMembraneCoordinatesY.data
		cl.enqueue_copy_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesX.data,self.dev_previousInterpolatedMembraneCoordinatesX.data).wait()
		cl.enqueue_copy_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesY.data,self.dev_previousInterpolatedMembraneCoordinatesY.data).wait()

		cl.enqueue_copy_buffer(self.queue,self.dev_contourCenter.data,self.dev_previousContourCenter.data).wait()

		#~ ipdb.set_trace()

		# set variable to tell host program that the tracking iteration has finished
		self.prg.setIterationFinished(self.queue, (1,1), None, self.dev_iterationFinished.data)
		barrierEvent = cl.enqueue_barrier(self.queue)
		cl.enqueue_read_buffer(self.queue, self.dev_iterationFinished.data, self.iterationFinished).wait()
		
		### debugging stuff ###
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarRadius.data, self.host_membranePolarRadius).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembranePolarRadius.data, self.host_interpolatedMembranePolarRadius).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
		#~ 
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesX.data, self.host_previousInterpolatedMembraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_previousInterpolatedMembraneCoordinatesY.data, self.host_previousInterpolatedMembraneCoordinatesY).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
		
		#~ print "self.dev_contourCenter:"
		#~ print self.dev_contourCenter
		#~ print "self.trackingFinished:"
		#~ print self.trackingFinished
		
		#~ if self.nrOfTrackingIterations>200:
		#~ if self.nrOfTrackingIterations>=1:
			#~ self.iterationFinished = 1
			#~ print "Execution time: "+str(self.getExectionTime())+" sec"
			#~ ipdb.set_trace()
		
		#~ ipdb.set_trace()
		#~ if self.trackingFinished:
		#~ if self.getContourId() == 5:
			#~ ipdb.set_trace()
			#~ self.getContourId()
			
			#~ plt.plot(self.host_membranePolarTheta,self.host_membranePolarRadius,'r')
			#~ plt.plot(self.host_interpolationAngles,self.host_interpolatedMembranePolarRadius,'b')
			#~ plt.show()
			
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY,'r')
			#~ plt.plot(self.host_previousInterpolatedMembraneCoordinatesX,self.host_previousInterpolatedMembraneCoordinatesY,'b')
			#~ plt.show()
			
			#~ plt.imshow(self.host_Img)
			#~ ax = plt.gca()
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,'k')
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY,'k')
			#~ x=self.host_membraneCoordinatesX
			#~ y=self.host_membraneCoordinatesY
			#~ pointIndexes = range(self.host_membraneCoordinatesX.shape[0])
			#~ z=pointIndexes
			#~ index = 990
			#~ plt.text(x[index],y[index],str(z[index]))
			#~ plt.show()


			#~ plt.imshow(self.host_Img)
			#~ ax = plt.gca()
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY,'k')
			#~ ax.invert_yaxis() # needed so that usage of 'plt.quiver' (below), will play along
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY,'r')
			#~ normalVectorScalingFactor = 3e-2
			#~ plt.quiver(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY,self.host_membraneNormalVectorsX,self.host_membraneNormalVectorsY,units='dots', pivot='tail',scale=normalVectorScalingFactor) #, pivot='middle'
			#~ pointIndexes = range(self.host_membraneCoordinatesX.shape[0])
			#~ z=pointIndexes
			#~ x=self.host_membraneCoordinatesX
			#~ y=self.host_membraneCoordinatesY
			#~ index = 990
			#~ plt.text(x[index],y[index],str(z[index]))
			#~ plt.show()
			
			
			#~ plt.imshow(self.host_Img)
			#~ ax = plt.gca()
			#~ ax.invert_yaxis() # needed so that usage of 'plt.quiver' (below), will play along
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,'r')
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY,'b')
			#~ normalVectorScalingFactor = 3e-2
			#~ plt.show()

		#cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
		#cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		#plt.imshow(self.host_Img)
		#ax = plt.gca()
		#ax.invert_yaxis() # needed so that usage of 'plt.quiver' (below), will play along
		#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,'r')
		#plt.show()
		
		pass
		
		#~ self.dev_trackingFinished = cl_array.to_device(self.ctx, self.queue, self.host_trackingFinished)
		#~ self.dev_iterationFinished = cl_array.to_device(self.ctx, self.queue, self.host_iterationFinished)
		
	def checkTrackingFinished(self):
		if self.nrOfTrackingIterations < self.minNrOfTrackingIterations:
			self.trackingFinished = 0 # force another iterations
		if self.nrOfTrackingIterations >= self.maxNrOfTrackingIterations:
			self.trackingFinished = 1 # force finish
		return self.trackingFinished
		pass
		
	def getMembraneCoordinatesX(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
		#~ return self.host_membraneCoordinatesX
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		return self.host_interpolatedMembraneCoordinatesX
		pass
	
	def getMembraneCoordinatesY(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		#~ return self.host_membraneCoordinatesY
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
		return self.host_contourCenter
		pass