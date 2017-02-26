import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
#~ import cv2 # OpenCV 2.3.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import Image
import json
from scipy import signal
#~ import ConfigParser
import ipdb
import os

class sequentialContourTracker( object ):
	def __init__(self, ctx, config):
		self.setupClQueue(ctx)
		self.loadClKernels()
		self.loadConfig(config)
		self.setupTrackingParameters()
		self.setWorkGroupSizes()
		self.setupClTrackingVariables()
		#~ self.loadBackground()
		
		# tracking status variables
		self.trackingFinished = np.int32(1) # True
		self.iterationFinished = np.int32(1) # True
		pass
	
	def setupClQueue(self,ctx):
		#~ self.ctx = cl.ctx([device])
		self.ctx = ctx
		#~ self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
		self.queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
		self.mf = cl.mem_flags
		
		#~ self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.NONE,cl.filter_mode.LINEAR)
		self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)
		pass
	
	def loadDarkfield(self, darkfieldList):
		if darkfieldList == []:
			self.darkfieldData = None
		else:
			for darkfieldIndex, darkfieldFile in enumerate(darkfieldList):
				#~ ipdb.set_trace()
				dark = Image.open(darkfieldFile)
				darkData = list(dark.getdata())
				if darkfieldIndex == 0:
					darkfieldDataSum = np.asarray(darkData, dtype=np.float32).reshape(dark.size)
				else:
					darkfieldDataSum = darkfieldDataSum + np.asarray(darkData, dtype=np.float32).reshape(dark.size)
		
			nrOfDarkfieldFiles = darkfieldList.__len__()
			self.darkfieldData = darkfieldDataSum/nrOfDarkfieldFiles # calculate mean darkfield
		pass
		
	def loadBackground(self, backgroundList):
		if backgroundList == []:
			self.backgroundData = None
		else:
			for backgroundIndex, backgroundFile in enumerate(backgroundList):
				bkgr = Image.open(backgroundFile)
				bkgrdata = list(bkgr.getdata())
				bkgrdataArray = np.asarray(bkgrdata, dtype=np.float32).reshape(bkgr.size)
				if self.darkfieldData is not None:
					bkgrdataArray = bkgrdataArray - self.darkfieldData

				if backgroundIndex == 0:
					backgroundDataSum = bkgrdataArray
				else:
					backgroundDataSum = backgroundDataSum + bkgrdataArray
			
			nrOfBackgroundFiles = backgroundList.__len__()
			self.backgroundData = backgroundDataSum/nrOfBackgroundFiles # calculate mean background
		pass
		
	def loadImage(self, imagePath):
		im = Image.open(imagePath)
		imgdata = list(im.getdata())
		
		imshape = im.size;
		imageData = np.asarray(imgdata, dtype=np.float32).reshape((imshape[1],imshape[0]))
		
		if self.darkfieldData is not None:
			imageData = imageData - self.darkfieldData
			
		if self.backgroundData is None:
			self.host_Img = imageData
		else:
			self.host_Img = imageData/self.backgroundData
		
		self.host_ImgUnfilteredUnscaled = self.host_Img
		
		if self.performImageFiltering is True:
			self.filterImage()
		
		if self.performImageScaling is True:
			self.rescaleImage()
		
		self.loadImageToGpu()
		pass
	
	def loadImageToGpu(self):
		self.dev_Img = cl.image_from_array(self.ctx, ary=self.host_Img, mode="r", norm_int=False, num_channels=1)
	
	def filterImage(self):
		#~ ipdb.set_trace()
		
		imageOrig = self.host_Img
		#~ imageNew = self.imageFilter(imageOrig,*self.filterArguments)
		imageNew = self.imageFilter(imageOrig)
		#~ plt.matshow(imageOrig)
		#~ plt.matshow(imageNew), plt.show()

		self.host_Img = np.array(imageNew,dtype=np.float32)
		
		pass
		
	def rescaleImage(self):
		# get image data into image structure for manipulations
		imgShape = np.asarray(self.host_ImgUnfilteredUnscaled.shape, dtype=np.float32)
		#~ imgShape = np.asarray(self.host_Img.shape, dtype=np.float32)
		im = Image.fromarray(self.host_Img)
		
		#~ newImageShape = tuple(np.int32(np.round(self.scalingFactor * imgShape)))
		imgShapeTmp = np.asarray([imgShape[1],imgShape[0]], dtype=np.float32)
		#~ imgShapeTmp = imgShape
		newImageShape = tuple(np.int32(np.round(self.scalingFactor * imgShapeTmp)))
		
		#~ newImage = im.resize(newImageShape, Image.NEAREST)
		newImage = im.resize(newImageShape, self.scalingMethodVar)
		
		# read back manipulated image data to host image array
		imgDataTmp = list(newImage.getdata())
		self.host_Img = np.asarray(imgDataTmp, dtype=np.float32).reshape((newImageShape[1],newImageShape[0]))
		
		#~ ipdb.set_trace()
		#~ plt.matshow(self.host_ImgUnfilteredUnscaled)
		#~ plt.matshow(newImage)
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
	
	def setImageScalingMethod(self,scalingMethod):
		if scalingMethod == "BICUBIC":
			scalingMethodVar = Image.BICUBIC
		if scalingMethod == "BILINEAR":
			scalingMethodVar = Image.BILINEAR
		if scalingMethod == "NEAREST":
			scalingMethodVar = Image.NEAREST
		if scalingMethod == "ANTIALIAS":
			scalingMethodVar = Image.ANTIALIAS
		
		return scalingMethodVar
		pass
	
	def setupWienerFilter(self,config):
		#~ filterSettings['filterkernelsize']
		#~ ipdb.set_trace()
		filterKernelSize = json.loads(config.get("ImageFilterParameters","filterKernelSize"))
		if filterKernelSize == "None":
			filterKernelSize = None
		else:
			filterKernelSize = json.loads(config.get("ImageFilterParameters","filterKernelSize"))
		
		noisePowerEstimate = config.get("ImageFilterParameters","noisePowerEstimate")
		if noisePowerEstimate == "None":
			noisePowerEstimate = None
			#~ ipdb.set_trace()
			#~ if self.snrRoi is not None:
				#~ noisePowerEstimate = "estimateFromSnrRoi"
		else:
			noisePowerEstimate = json.loads(config.get("ImageFilterParameters","noisePowerEstimate"))
		
		#~ self.imageFilter = signal.wiener
		self.imageFilter = self.wienerFilterMod
		self.filterArguments = (filterKernelSize,noisePowerEstimate)
		pass
		
	def wienerFilterMod(self,imageData):
		kernelSize = self.filterArguments[0]
		if self.filterArguments[1] == "estimateFromSnrRoi":
			noisePower = self.getImageStd()**2
		else:
			noisePower = self.filterArguments[1]
		filterArgs = (kernelSize,noisePower)
		return signal.wiener(imageData,*filterArgs)
		
	def setupFilter(self,config):
		#~ ipdb.set_trace()
		filterType = json.loads(config.get("ImageFilterParameters","filterType"))
		
		if filterType == "wiener":
			self.setupWienerFilter(config)
		pass
	
	def loadConfig(self,config):
	# from: http://stackoverflow.com/questions/335695/lists-in-configparser
	# json.loads(self.config.get("SectionOne","startingCoordinate"))
		snrRoi = config.get("TrackingParameters","snrRoi")
		if snrRoi == "" or snrRoi == "None":
			self.snrRoi = None
		else:
			self.snrRoi = np.array(json.loads(config.get("TrackingParameters","snrRoi")))
		
		performImageFiltering = config.get("ImageFilterParameters","performImageFiltering")
		if performImageFiltering == "True":
			self.performImageFiltering = True
		else:
			self.performImageFiltering = False
		
		if self.performImageFiltering:
			#~ filterSettings = config._sections['ImageFilterParameters']
			self.setupFilter(config)
			
		performImageScaling = config.get("ImageManipulationParameters","performImageScaling")
		
		if performImageScaling == "True":
			self.performImageScaling = True
		else:
			self.performImageScaling = False
		
		if self.performImageScaling == True:
			self.scalingFactor = np.float64(json.loads(config.get("ImageManipulationParameters","scalingFactor")))
		else:
			self.scalingFactor = np.float64(1)

		self.scalingMethod = json.loads(config.get("ImageManipulationParameters","scalingMethod"))
		self.scalingMethodVar = self.setImageScalingMethod(self.scalingMethod)
		
		self.startingCoordinate = self.scalingFactor * np.array(json.loads(config.get("TrackingParameters","startingCoordinate")))
		self.rotationCenterCoordinate = self.scalingFactor * np.array(json.loads(config.get("TrackingParameters","rotationCenterCoordinate")))
		#~ self.membraneNormalVector = np.array(json.loads(config.get("TrackingParameters","membraneNormalVector")))
		
		self.linFitParameter = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","linFitParameter")))
		self.linFitSearchRange = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","linFitSearchRange")))
		#~ self.interpolationFactor = json.loads(config.get("TrackingParameters","interpolationFactor"))
		#~ self.interpolationFactor = np.int32(np.float64(json.loads(config.get("TrackingParameters","interpolationFactor"))))
		self.interpolationFactor = np.int32(np.float64(json.loads(config.get("TrackingParameters","interpolationFactor")))/self.scalingFactor)

		self.meanParameter = np.int32(np.round(self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","meanParameter")))))
		self.meanRangePositionOffset = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","meanRangePositionOffset")))
		
		self.localAngleRange = np.float64(json.loads(config.get("TrackingParameters","localAngleRange")))
		self.nrOfLocalAngleSteps = np.int32(json.loads(config.get("TrackingParameters","nrOfLocalAngleSteps")))
		
		self.detectionKernelStrideSize = np.int32(json.loads(config.get("TrackingParameters","detectionKernelStrideSize")))
		self.nrOfStrides = np.int32(json.loads(config.get("TrackingParameters","nrOfStrides")))
		
		self.nrOfIterationsPerContour = np.int32(json.loads(config.get("TrackingParameters","nrOfIterationsPerContour")))
		
		self.inclineTolerance = np.float64(json.loads(config.get("TrackingParameters","inclineTolerance")))

		self.inclineRefinementRange = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","inclineRefinementRange")))
		
		#~ backgroundImagePath = config.get("FileParameters","backgroundImagePath")
		#~ if backgroundImagePath == "" or backgroundImagePath == "None":
			#~ self.backgroundImagePath = None
		#~ else:
			#~ self.backgroundImagePath = json.loads(backgroundImagePath)
		
		self.computeDeviceId = json.loads(config.get("OpenClParameters","computeDeviceId"))
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
		initialNormalVector = self.startingCoordinate - self.rotationCenterCoordinate
		initialNormalVectorNorm = np.sqrt(initialNormalVector[0]**2 + initialNormalVector[1]**2)
		#~ tmp = initialNormalVector/initialNormalVectorNorm
		#~ self.radiusUnitVector = np.array([tmp[1],tmp[0]])
		self.radiusUnitVector = initialNormalVector/initialNormalVectorNorm
		
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

		self.inclineRefinementRange = self.inclineRefinementRange*self.interpolationFactor/2
		self.inclineRefinementRange = np.int32(self.inclineRefinementRange)
		
		pass

	def setupClTrackingVariables(self):
		self.host_fitIncline = np.empty(self.nrOfLocalAngleSteps,dtype=np.float64)
		self.dev_fitIncline = cl_array.to_device(self.queue, self.host_fitIncline)

		self.host_fitIntercept = np.empty(self.nrOfLocalAngleSteps,dtype=np.float64)
		self.dev_fitIntercept = cl_array.to_device(self.queue, self.host_fitIntercept)
		
		self.host_localMembranePositionsX = np.zeros(shape=self.nrOfLocalAngleSteps,dtype=np.float64)
		#~ host_localMembranePositionsX = np.zeros(shape=self.localAngles.shape[0],dtype=np.float64)
		self.dev_localMembranePositionsX = cl_array.to_device(self.queue, self.host_localMembranePositionsX)

		self.host_localMembranePositionsY = np.zeros(shape=self.nrOfLocalAngleSteps,dtype=np.float64)
		#~ host_localMembranePositionsY = np.zeros(shape=localAngles.shape[0],dtype=np.float64)
		self.dev_localMembranePositionsY = cl_array.to_device(self.queue, self.host_localMembranePositionsY)

		self.host_membraneCoordinatesX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneCoordinatesX[0] = self.startingCoordinate[0]
		self.dev_membraneCoordinatesX = cl_array.to_device(self.queue, self.host_membraneCoordinatesX)
		
		self.host_membraneCoordinatesY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.host_membraneCoordinatesY[0] = self.startingCoordinate[1]
		self.dev_membraneCoordinatesY = cl_array.to_device(self.queue, self.host_membraneCoordinatesY)

		self.host_fitInclines = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_fitInclines = cl_array.to_device(self.queue, self.host_fitInclines)


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

		self.host_membranePolarRadius = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarRadius = cl_array.to_device(self.queue, self.host_membranePolarRadius)
		self.dev_membranePolarRadiusTMP = cl_array.to_device(self.queue, self.host_membranePolarRadius)

		self.host_membranePolarTheta = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarTheta = cl_array.to_device(self.queue, self.host_membranePolarTheta)
		self.dev_membranePolarThetaTMP = cl_array.to_device(self.queue, self.host_membranePolarTheta)
		
		self.host_membranePolarRadiusInterpolation = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarRadiusInterpolation = cl_array.to_device(self.queue, self.host_membranePolarRadiusInterpolation)
		self.dev_membranePolarRadiusInterpolationTesting = cl_array.to_device(self.queue, self.host_membranePolarRadiusInterpolation)

		self.host_membranePolarThetaInterpolation = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_membranePolarThetaInterpolation = cl_array.to_device(self.queue, self.host_membranePolarThetaInterpolation)
		self.dev_membranePolarThetaInterpolationTesting = cl_array.to_device(self.queue, self.host_membranePolarThetaInterpolation)

		#~ host_interpolationAngles = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		#~ host_interpolationAngles = angleStepSize*np.float64(range(self.nrOfDetectionAngleSteps))
		self.host_interpolationAngles = np.float64(np.linspace(-np.pi,np.pi,self.nrOfDetectionAngleSteps))
		self.dev_interpolationAngles = cl_array.to_device(self.queue, self.host_interpolationAngles)

		self.host_b = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_b = cl_array.to_device(self.queue, self.host_b)

		self.host_c = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_c = cl_array.to_device(self.queue, self.host_c)

		self.host_d = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_d = cl_array.to_device(self.queue, self.host_d)

		self.host_dbgOut = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_dbgOut = cl_array.to_device(self.queue, self.host_dbgOut)

		########################################################################
		### setup OpenCL local memory
		########################################################################
		self.localMembranePositions_memSize = self.dev_localMembranePositionsX.nbytes
		self.fitIncline_memSize = self.dev_fitIncline.nbytes
		self.fitIntercept_memSize = self.dev_fitIntercept.nbytes
		self.membranePolarThetaInterpolation_memSize = self.dev_membranePolarThetaInterpolation.nbytes
		self.membranePolarRadiusInterpolation_memSize = self.dev_membranePolarRadiusInterpolation.nbytes

		self.host_membraneNormalVectors = np.zeros(self.nrOfDetectionAngleSteps, cl.array.vec.double2)
		self.dev_membraneNormalVectors = cl_array.to_device(self.queue, self.host_membraneNormalVectors)
		self.membraneNormalVectors_memSize = self.dev_membraneNormalVectors.nbytes

		self.host_contourCenter = np.zeros(1, cl.array.vec.double2)
		self.dev_contourCenter = cl_array.to_device(self.queue, self.host_contourCenter)
		
		self.host_rotatedUnitVector = np.zeros(self.nrOfLocalAngleSteps, cl.array.vec.double2)
		self.dev_rotatedUnitVector = cl_array.to_device(self.queue, self.host_rotatedUnitVector)
		self.rotatedUnitVector_memSize = self.dev_rotatedUnitVector.nbytes * int(self.contourPointsPerWorkGroup)
		
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
	
	def trackContour(self):
		#~ ipdb.set_trace()
		for coordinateIndex in xrange(int(self.nrOfDetectionAngleSteps)):
			#~ print coordinateIndex
			
			coordinateIndex = np.int32(coordinateIndex)
			
			angle = self.angleStepSize*np.float64(coordinateIndex+1)
			
			radiusVectorRotationMatrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
			
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
										 #coordinateIndex).wait()
			
			self.prg.findMembranePositionNew2(self.queue, self.global_size, self.local_size, self.sampler, \
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
											 self.dev_fitInclines.data, \
											 coordinateIndex, \
											 self.inclineTolerance, \
											 self.inclineRefinementRange)
											 
			#self.prg.findMembranePositionNew3(self.queue, self.global_size, self.local_size, self.sampler, \
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
											 #coordinateIndex, \
											 #self.inclineTolerance, \
											 #self.meanIntensity, \
											 #self.inclineRefinementRange \
											 #)
			
			cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()

			cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()

			####### DEBUG SECTION START #######
			##~ if coordinateIndex == 2047:
				##~ plt.imshow(self.host_Img)
				##~ ax = plt.gca()
				##~ ax.invert_yaxis() # needed so that usage of 'plt.quiver' (below), will play along
				##~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,'k')
				##~ plt.show()

			#cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
			#if coordinateIndex >= 2047:
				#print str(coordinateIndex)
			##~ if coordinateIndex%100 is true:
				#plt.imshow(self.host_Img)
				#ax = plt.gca()
				#ax.invert_yaxis() # needed so that usage of 'plt.quiver' (below), will play along
				#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,'k')
				#normalVectorScalingFactor = 3e-2
				#plt.quiver(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,self.host_membraneNormalVectorsX,self.host_membraneNormalVectorsY,units='dots', pivot='tail',scale=normalVectorScalingFactor) #, pivot='middle'
				#plt.show()

			##~ ipdb.set_trace()
			####### DEBUG SECTION STOP #######
			
			#~ cl.enqueue_read_buffer(queue, dev_membraneNormalVectorsX.data, host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(queue, dev_membraneNormalVectorsY.data, host_membraneNormalVectorsY).wait()
				
			currentMembraneCoordinate = np.array([self.host_membraneCoordinatesX[coordinateIndex],self.host_membraneCoordinatesY[coordinateIndex]])
			
			radiusVector = currentMembraneCoordinate - self.rotationCenterCoordinate
			radiusVectorNorm = np.sqrt(radiusVector[0]**2 + radiusVector[1]**2)
			
			#~ ipdb.set_trace()
			#~ radiusUnitVector = radiusVector/radiusVectorNorm
			#~ print radiusUnitVector
			#~ print radiusVectorNorm
			
			rotatedRadiusUnitVector = radiusVectorRotationMatrix.dot(self.radiusUnitVector)
			
			nextMembranePosition = self.rotationCenterCoordinate + rotatedRadiusUnitVector*radiusVectorNorm
			#~ nextMembraneNormalVector = rotatedRadiusUnitVector
			#~ ipdb.set_trace()
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
		self.prg.calculateMembraneNormalVectors(self.queue, self.gradientGlobalSize, None, \
										   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
										   self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data \
										   #~ cl.LocalMemory(membraneNormalVectors_memSize) \
										  )
		
		self.prg.calculateDs(self.queue, self.gradientGlobalSize, None, \
			   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
			   self.dev_ds.data \
			 )
			 
		self.prg.calculateSumDs(self.queue, self.gradientGlobalSize, None, \
			   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
			   self.dev_ds.data, self.dev_sumds.data \
			 )
			 
		self.prg.calculateContourCenterNew2(self.queue, (1,1), None, \
								   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								   self.dev_ds.data, self.dev_sumds.data, \
								   self.dev_contourCenter.data, \
								   np.int32(self.nrOfDetectionAngleSteps) \
								  )		
			 
		self.queue.finish()

	def getMembraneCoordinatesX(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
		#~ return self.host_membraneCoordinatesX
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		return self.host_membraneCoordinatesX/self.scalingFactor
		pass

	def getMembraneCoordinatesY(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		#~ return self.host_membraneCoordinatesY
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		return self.host_membraneCoordinatesY/self.scalingFactor
		pass

	def getMembraneCoordinatesXscaled(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
		#~ return self.host_membraneCoordinatesX
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		return self.host_membraneCoordinatesX
		pass

	def getMembraneCoordinatesYscaled(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		#~ return self.host_membraneCoordinatesY
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		return self.host_membraneCoordinatesY
		pass

	def getFitInclines(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		#~ return self.host_membraneCoordinatesY
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		cl.enqueue_read_buffer(self.queue, self.dev_fitInclines.data, self.host_fitInclines).wait()
		return self.host_fitInclines * self.scalingFactor # needs to be multiplied, since putting in more pixels artificially reduces the value of the incline
		pass

	def getSnrRoiScaled(self):
		return np.floor(self.snrRoi*self.scalingFactor)
		pass
	
	def getSnrRoi(self):
		return self.snrRoi
		pass
	
	def getImageSnr(self):
		roiStd = self.getImageStd()
		roiMean = self.getImageIntensity()
		return roiMean/roiStd
		pass

	def getImageStd(self):
		roiValues = self.getRoiIntensityValues()
		return roiValues.std()
		
	def getImageIntensity(self):
		roiValues = self.getRoiIntensityValues()
		return roiValues.mean()
		pass

	def getRoiIntensityValues(self):
		snrRoi = self.getSnrRoi()
		snrRoiStartIndexes = snrRoi[0]
		snrRoiStopIndexes = snrRoi[1]
		#~ return self.host_Img[snrRoiStartIndexes[0]:snrRoiStopIndexes[1],snrRoiStartIndexes[0]:snrRoiStopIndexes[1]]
		return self.host_ImgUnfilteredUnscaled[snrRoiStartIndexes[0]:snrRoiStopIndexes[1],snrRoiStartIndexes[0]:snrRoiStopIndexes[1]]		
		
