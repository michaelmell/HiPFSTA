import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import cv2 # OpenCV 2.3.1
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Image
import json
from scipy import signal
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
		#~ self.loadBackground()
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
		imgShape = np.asarray(self.host_Img.shape, dtype=np.float32)
		im = Image.fromarray(self.host_Img)
		
		newImageShape = tuple(np.int32(np.round(self.scalingFactor * imgShape)))
		
		#~ newImage = im.resize(newImageShape, Image.NEAREST)
		newImage = im.resize(newImageShape, self.scalingMethodVar)
		
		# read back manipulated image data to host image array
		imgDataTmp = list(newImage.getdata())
		self.host_Img = np.asarray(imgDataTmp, dtype=np.float32).reshape((newImageShape[1],newImageShape[0]))
		
		#~ plt.matshow(self.host_Img)
		#~ plt.show()
		#~ ipdb.set_trace()

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
		snrRoi = config.get("FileParameters","snrRoi")
		if snrRoi == "" or snrRoi == "None":
			self.snrRoi = None
		else:
			self.snrRoi = np.array(json.loads(config.get("FileParameters","snrRoi")))
		
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
		self.membraneNormalVector = np.array(json.loads(config.get("TrackingParameters","membraneNormalVector")))
		
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
		
		self.nrOfAnglesToCompare = np.int32(json.loads(config.get("TrackingParameters","nrOfAnglesToCompare")))
		
		self.nrOfIterationsPerContour = np.int32(json.loads(config.get("TrackingParameters","nrOfIterationsPerContour")))
		
		#~ backgroundImagePath = config.get("FileParameters","backgroundImagePath")
		#~ if backgroundImagePath == "" or backgroundImagePath == "None":
			#~ self.backgroundImagePath = None
		#~ else:
			#~ self.backgroundImagePath = json.loads(backgroundImagePath)
		
		self.computeDeviceId = json.loads(config.get("OpenClParameters","computeDeviceId"))
		
		self.coordinateTolerance = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","coordinateTolerance")))
		
		self.maxNrOfTrackingIterations = json.loads(config.get("TrackingParameters","maxNrOfTrackingIterations"))
		self.minNrOfTrackingIterations = json.loads(config.get("TrackingParameters","minNrOfTrackingIterations"))
		
		self.inclineTolerance = np.float64(json.loads(config.get("TrackingParameters","inclineTolerance")))
		
		#~ self.intensityRoiTopLeft = json.loads(config.get("TrackingParameters","intensityRoiTopLeft"))
		#~ self.intensityRoiBottomRight = json.loads(config.get("TrackingParameters","intensityRoiBottomRight"))
		
		self.centerTolerance = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","centerTolerance")))
		
		self.maxInterCoordinateAngle = np.float64(json.loads(config.get("TrackingParameters","maxInterCoordinateAngle")))
		self.maxCoordinateShift = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","maxCoordinateShift")))
		
		resetNormalsAfterEachImage = config.get("TrackingParameters","resetNormalsAfterEachImage")
		if resetNormalsAfterEachImage == 'True':
			self.resetNormalsAfterEachImage = True
		else:
			self.resetNormalsAfterEachImage = False

		#~ self.minAngleDifference = np.float64(json.loads(config.get("TrackingParameters","minAngleDifference")))

		self.inclineRefinementRange = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","inclineRefinementRange")))
		
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

		self.inclineRefinementRange = self.inclineRefinementRange*self.interpolationFactor/2
		self.inclineRefinementRange = np.int32(self.inclineRefinementRange)
		
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
		
		self.host_closestLowerNoneNanIndex = np.int32(range(0,np.float32(self.nrOfDetectionAngleSteps)))
		self.dev_closestLowerNoneNanIndex = cl_array.to_device(self.queue, self.host_closestLowerNoneNanIndex)
		
		self.host_closestUpperNoneNanIndex = np.int32(range(0,np.float32(self.nrOfDetectionAngleSteps)))
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

		#~ host_interpolationAngles = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		#~ host_interpolationAngles = angleStepSize*np.float64(range(self.nrOfDetectionAngleSteps))
		startAngle = -np.pi
		endAngle = np.pi - 2*np.pi/self.nrOfDetectionAngleSteps # we substract, so that we have no angle overlap
		#~ self.host_interpolationAngles = np.float64(np.linspace(-np.pi,np.pi,self.nrOfDetectionAngleSteps))
		self.host_interpolationAngles = np.float64(np.linspace(startAngle,endAngle,self.nrOfDetectionAngleSteps))
		self.dev_interpolationAngles = cl_array.to_device(self.queue, self.host_interpolationAngles)
		
		self.host_b = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_b = cl_array.to_device(self.queue, self.host_b)

		self.host_c = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_c = cl_array.to_device(self.queue, self.host_c)

		self.host_d = np.zeros(shape=2*self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_d = cl_array.to_device(self.queue, self.host_d)

		self.host_dbgOut = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_dbgOut = cl_array.to_device(self.queue, self.host_dbgOut)
		
		self.host_dbgOut2 = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_dbgOut2 = cl_array.to_device(self.queue, self.host_dbgOut)
		
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
		self.dev_membraneNormalVectors = cl_array.to_device(self.queue, self.host_membraneNormalVectors)
		self.membraneNormalVectors_memSize = self.dev_membraneNormalVectors.nbytes
		
		#~ self.host_fitIncline = np.empty(self.nrOfLocalAngleSteps,dtype=np.float64)
		self.host_rotatedUnitVector = np.zeros(self.nrOfLocalAngleSteps, cl.array.vec.double2)
		self.dev_rotatedUnitVector = cl_array.to_device(self.queue, self.host_rotatedUnitVector)
		self.rotatedUnitVector_memSize = self.dev_rotatedUnitVector.nbytes * int(self.contourPointsPerWorkGroup)
		
		self.host_contourCenter = np.zeros(1, cl.array.vec.double2)
		self.dev_contourCenter = cl_array.to_device(self.queue, self.host_contourCenter)
		self.dev_previousContourCenter = cl_array.to_device(self.queue, self.host_contourCenter)
		
		self.host_listOfGoodCoordinates = np.zeros(self.nrOfDetectionAngleSteps, dtype=np.int32)
		self.dev_listOfGoodCoordinates = cl_array.to_device(self.queue, self.host_listOfGoodCoordinates)
		self.listOfGoodCoordinates_memSize = self.dev_listOfGoodCoordinates.nbytes
		#~ self.trackingFinished = np.array(1,dtype=np.int32) # True
		
		### setup radial vectors used for linear interpolation
		self.host_radialVectors = np.zeros(shape=self.nrOfDetectionAngleSteps, dtype=cl.array.vec.double2)
		self.host_radialVectorsX = np.zeros(shape=self.nrOfDetectionAngleSteps, dtype=np.float64)
		self.host_radialVectorsY = np.zeros(shape=self.nrOfDetectionAngleSteps, dtype=np.float64)
		
		#~ self.host_interpolationAngles = np.float64(np.linspace(startAngle,endAngle,self.nrOfDetectionAngleSteps))
		#~ self.dev_interpolationAngles = cl_array.to_device(self.queue, self.host_interpolationAngles)

		
		#~ angle = self.angleStepSize*np.float64(coordinateIndex+1)
		#~ radiusUnitVector = np.array([1,0])
		radiusUnitVector = np.array([1,0],dtype=np.float64)
		
		for counter, angle in enumerate(self.host_interpolationAngles):
			radiusVectorRotationMatrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
			#~ ipdb.set_trace()
			rotatedRadiusUnitVector = radiusVectorRotationMatrix.dot(radiusUnitVector)
			self.host_radialVectors[counter] = rotatedRadiusUnitVector
			self.host_radialVectorsX[counter] = rotatedRadiusUnitVector[0]
			self.host_radialVectorsY[counter] = rotatedRadiusUnitVector[1]
		
		self.dev_radialVectors = cl_array.to_device(self.queue, self.host_radialVectors)
		self.dev_radialVectorsX = cl_array.to_device(self.queue, self.host_radialVectorsX)
		self.dev_radialVectorsY = cl_array.to_device(self.queue, self.host_radialVectorsY)

		#~ ipdb.set_trace()
		#~ zeros = np.zeros(shape=self.nrOfDetectionAngleSteps)
		#~ plt.quiver(zeros,zeros,self.host_radialVectors[:]['x'],self.host_radialVectors[:]['y'],minlength=100), plt.show()
		
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
		if self.resetNormalsAfterEachImage and not self.getContourId()==0: # reset contour normal vector to radial vectors; we do this only starting for the second, since doing this for image 0, would destroy the correspondence of the indexes of the contour coordinates to their corresponding contour normals
			cl.enqueue_copy_buffer(self.queue,self.dev_radialVectorsX.data,self.dev_membraneNormalVectorsX.data).wait()
			cl.enqueue_copy_buffer(self.queue,self.dev_radialVectorsY.data,self.dev_membraneNormalVectorsY.data).wait()
		else: # copy contour normal vectors from last image to use as initial normal vectors for next image
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
		self.dev_trackingFinished = cl_array.to_device(self.queue, self.trackingFinished)
		
		#~ self.iterationFinished = np.int32(0) # True
		self.iterationFinished = np.array(0,dtype=np.int32) # True
		self.dev_iterationFinished = cl_array.to_device(self.queue, self.iterationFinished)
		
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

		##~ self.dev_membraneCoordinatesX = cl_array.to_device(self.queue, self.host_membraneCoordinatesX)
		
		#self.queue.finish()
		
		#~ print "xCoord[0]:"+str(self.host_membraneCoordinatesXdebug[0])
		#~ print "yCoord[0]:"+str(self.host_membraneCoordinatesYdebug[0])
		
		#~ if self.getContourId() >= 5598 and self.getNrOfTrackingIterations() >= 1:
			#~ print "Iteration: "+str(self.getNrOfTrackingIterations())
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_contourCenter.data,self.host_contourCenter).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesX.data,self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesY.data,self.host_interpolatedMembraneCoordinatesY).wait()
			#~ ind = 805
			#~ x = self.host_contourCenter['x'][0]
			#~ y = self.host_contourCenter['y'][0]
			#~ dx = self.host_interpolatedMembraneCoordinatesX[ind] - x
			#~ dy = self.host_interpolatedMembraneCoordinatesY[ind] - y
			#~ plt.arrow( x, y, dx, dy, fc="k" )
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY), plt.show()
#~ 
			#~ x = self.host_contourCenter['x'][0]
			#~ y = self.host_contourCenter['y'][0]
			#~ dx = self.host_membraneCoordinatesX[ind] - x
			#~ dy = self.host_membraneCoordinatesY[ind] - y
			#~ plt.arrow( x, y, dx, dy, fc="k" )
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
#~ 
			#~ x = self.host_contourCenter['x'][0]
			#~ y = self.host_contourCenter['y'][0]
			#~ dx = self.host_membraneCoordinatesX[ind] - x
			#~ dy = self.host_membraneCoordinatesY[ind] - y
			#~ plt.arrow( x, y, dx, dy, fc="k" )
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY,'r')
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
		#~ 
			#~ ipdb.set_trace()


		#~ if self.getContourId() >= 11598 and self.getNrOfTrackingIterations() >= 14:
			#~ print "Iteration: "+str(self.getNrOfTrackingIterations())
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_contourCenter.data,self.host_contourCenter).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesX.data,self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesY.data,self.host_interpolatedMembraneCoordinatesY).wait()
			#~ ind = 805
			#~ x = self.host_contourCenter['x'][0]
			#~ y = self.host_contourCenter['y'][0]
			#~ dx = self.host_interpolatedMembraneCoordinatesX[ind] - x
			#~ dy = self.host_interpolatedMembraneCoordinatesY[ind] - y
			#~ plt.arrow( x, y, dx, dy, fc="k" )
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY), plt.show()
#~ 
			#~ x = self.host_contourCenter['x'][0]
			#~ y = self.host_contourCenter['y'][0]
			#~ dx = self.host_membraneCoordinatesX[ind] - x
			#~ dy = self.host_membraneCoordinatesY[ind] - y
			#~ plt.arrow( x, y, dx, dy, fc="k" )
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
#~ 
			#~ x = self.host_contourCenter['x'][0]
			#~ y = self.host_contourCenter['y'][0]
			#~ dx = self.host_membraneCoordinatesX[ind] - x
			#~ dy = self.host_membraneCoordinatesY[ind] - y
			#~ plt.arrow( x, y, dx, dy, fc="k" )
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY,'r')
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
		#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembranePolarRadius.data,self.host_interpolatedMembranePolarRadius).wait()
			#~ plt.plot(self.host_interpolatedMembranePolarRadius), plt.show()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarRadius.data,self.host_membranePolarRadius).wait()
			#~ plt.plot(self.host_membranePolarRadius), plt.show()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarTheta.data,self.host_membranePolarTheta).wait()
			#~ plt.plot(self.host_membranePolarTheta), plt.show()
			#~ 
			#~ plt.plot(np.diff(self.host_membranePolarTheta)), plt.show()
#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolationAngles.data,self.host_interpolationAngles).wait()
			#~ plt.plot(self.host_interpolationAngles), plt.show()
#~ 
			#~ ipdb.set_trace()

		#~ if self.getContourId() >= 11598 and self.getNrOfTrackingIterations() >= 8:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut.data, self.host_dbgOut).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut2.data, self.host_dbgOut2).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
#~ 
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY)
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY)
			#~ plt.show()
#~ 
			#~ plt.plot(self.host_interpolationAngles)
			#~ plt.plot(self.host_dbgOut)
			#~ plt.plot(self.host_dbgOut2)
			#~ plt.show()
			#~ 
			#~ ipdb.set_trace()
		#~ ipdb.set_trace()
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
											 self.dev_fitInclines.data, \
											 kernelCoordinateStartingIndex, \
											 self.inclineTolerance, \
											 self.inclineRefinementRange )
											 
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
											 #self.meanIntensity, \
											 #self.inclineRefinementRange \
											 #)

		#~ print "Contour ID: "+str(self.getContourId())
		#~ print "Iteration: "+str(self.getNrOfTrackingIterations())
		#~ if self.getContourId() >= 130 and self.getNrOfTrackingIterations() >= 0:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsX.data, self.host_membraneNormalVectorsX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneNormalVectorsY.data, self.host_membraneNormalVectorsY).wait()
#~ 
			#~ plt.imshow(self.host_Img)
			#~ ax = plt.gca()
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,'k')
			#~ ax.invert_yaxis() # needed so that usage of 'plt.quiver' (below), will play along
			#~ normalVectorScalingFactor = 3e-2
			#~ plt.quiver(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY,self.host_membraneNormalVectorsX,self.host_membraneNormalVectorsY,units='dots', pivot='tail',scale=normalVectorScalingFactor) #, pivot='middle'
			#~ plt.show()
			#~ ipdb.set_trace()



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
		
		#~ if self.getContourId() >= 14 and self.getNrOfTrackingIterations() >= 0:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_contourCenter.data,self.host_contourCenter).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ ind = 805
			#~ x = self.host_contourCenter['x'][0]
			#~ y = self.host_contourCenter['y'][0]
			#~ dx = self.host_membraneCoordinatesX[ind] - x
			#~ dy = self.host_membraneCoordinatesY[ind] - y
			#~ plt.arrow( x, y, dx, dy, fc="k" )
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
#~ 
			#~ ipdb.set_trace()

		#~ if self.getContourId() >= 10140 and self.getNrOfTrackingIterations() == 1:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_contourCenter.data,self.host_contourCenter).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#~ ipdb.set_trace()
		
		#self.prg.filterNanValues(self.queue, self.gradientGlobalSize, None, \
								 ##~ cl.LocalMemory(self.localMembranePositions_memSize), cl.LocalMemory(self.localMembranePositions_memSize), \
								 #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								 #self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
								 #self.dev_closestLowerNoneNanIndex.data, self.dev_closestUpperNoneNanIndex.data \
								 #)
		self.prg.filterNanValues(self.queue, self.gradientGlobalSize, None, \
								 self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								 self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
								 #~ cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes) \
								 self.dev_closestLowerNoneNanIndex.data, self.dev_closestUpperNoneNanIndex.data \
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
		
		#~ if self.getContourId() >= 14 and self.getNrOfTrackingIterations() >= 0:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_contourCenter.data,self.host_contourCenter).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#~ ipdb.set_trace()
		
		#~ ipdb.set_trace()

		#~ cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interCoordinateAngles).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()

		#print self.getContourId()
		#print self.getNrOfTrackingIterations()
		
		#if self.getContourId() == 14825 and self.getNrOfTrackingIterations() >= 0:
			##~ self.host_interpolationAnglesOld = np.float64(np.zeros(2048))
			##~ cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interpolationAnglesOld).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			##~ plt.plot(self.host_interpolationAnglesOld), plt.show()
			#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#cl.enqueue_read_buffer(self.queue,self.dev_listOfGoodCoordinates.data,self.host_listOfGoodCoordinates).wait()
			#plt.plot(self.host_listOfGoodCoordinates), plt.ylim([-0.1,1.1]), plt.show()
			#ipdb.set_trace()

		#~ if self.getContourId() >= 14 and self.getNrOfTrackingIterations() >= 0:
			#~ print "before filterJumpedCoordinates"
			#~ cl.enqueue_read_buffer(self.queue,self.dev_contourCenter.data,self.host_contourCenter).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#~ ipdb.set_trace()

		
		#~ ipdb.set_trace()
		self.prg.filterJumpedCoordinates(self.queue, self.gradientGlobalSize, None, \
											self.dev_previousContourCenter.data, \
											self.dev_membraneCoordinatesX.data, \
											self.dev_membraneCoordinatesY.data, \
											self.dev_membraneNormalVectorsX.data, \
											self.dev_membraneNormalVectorsY.data, \
										    self.dev_previousInterpolatedMembraneCoordinatesX.data, \
											self.dev_previousInterpolatedMembraneCoordinatesY.data, \
										    #~ cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), \
											#~ cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
											#~ cl.LocalMemory(self.listOfGoodCoordinates_memSize), \
										    self.dev_closestLowerNoneNanIndex.data, \
											self.dev_closestUpperNoneNanIndex.data, \
											self.dev_listOfGoodCoordinates.data, \
											self.maxCoordinateShift, \
											self.dev_listOfGoodCoordinates.data \
											)
		
		#if self.getContourId() == 2447 and self.getNrOfTrackingIterations() >= 1:
			##~ self.host_interpolationAnglesOld = np.float64(np.zeros(2048))
			##~ cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interpolationAnglesOld).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			##~ plt.plot(self.host_interpolationAnglesOld), plt.show()
			#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#cl.enqueue_read_buffer(self.queue,self.dev_listOfGoodCoordinates.data,self.host_listOfGoodCoordinates).wait()
			#plt.plot(self.host_listOfGoodCoordinates), plt.ylim([-0.1,1.1]), plt.show()
			##~ ipdb.set_trace()

		barrierEvent = cl.enqueue_barrier(self.queue)

		#~ if self.getContourId() >= 14 and self.getNrOfTrackingIterations() >= 0:
			#~ print "after filterJumpedCoordinates"
			#~ cl.enqueue_read_buffer(self.queue,self.dev_contourCenter.data,self.host_contourCenter).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#~ ipdb.set_trace()

		self.prg.calculateInterCoordinateAngles(self.queue, self.gradientGlobalSize, None, \
												self.dev_interCoordinateAngles.data, \
												self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data \
											   )

		barrierEvent = cl.enqueue_barrier(self.queue)

		#if self.getContourId() == 2447 and self.getNrOfTrackingIterations() == 20:
			#self.host_interpolationAnglesOld = np.float64(np.zeros(2048))
			#cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interpolationAnglesOld).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#plt.plot(self.host_interpolationAnglesOld), plt.show()
			#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#cl.enqueue_read_buffer(self.queue,self.dev_listOfGoodCoordinates.data,self.host_listOfGoodCoordinates).wait()
			#plt.plot(self.host_listOfGoodCoordinates), plt.ylim([-0.1,1.1]), plt.show()
			#ipdb.set_trace()

		##~ self.prg.checkIfTrackingFinished(self.queue, self.gradientGlobalSize, None, \
										 ##~ self.dev_interpolatedMembraneCoordinatesX.data, \
										 ##~ self.dev_interpolatedMembraneCoordinatesY.data, \
										 ##~ self.dev_previousInterpolatedMembraneCoordinatesX.data, \
										 ##~ self.dev_previousInterpolatedMembraneCoordinatesY.data, \
										 ##~ self.dev_trackingFinished.data, \
										 ##~ self.coordinateTolerance)
										 		
		#if self.getContourId() == 2447 and self.getNrOfTrackingIterations() == 20:
			#self.host_interpolationAnglesOld = np.float64(np.zeros(2048))
			#cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interpolationAnglesOld).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#plt.plot(self.host_interpolationAnglesOld), plt.show()
			#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#cl.enqueue_read_buffer(self.queue,self.dev_listOfGoodCoordinates.data,self.host_listOfGoodCoordinates).wait()
			#plt.plot(self.host_listOfGoodCoordinates), plt.ylim([-0.1,1.1]), plt.show()
			#ipdb.set_trace()

		#if self.getContourId() == 2448:
			#cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interpolationAnglesOld).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#plt.plot(self.host_interCoordinateAngles), plt.show()
			#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#ipdb.set_trace()
		
		self.prg.filterIncorrectCoordinates(self.queue, self.gradientGlobalSize, None, \
											self.dev_previousContourCenter.data, \
										    self.dev_interCoordinateAngles.data, \
										    self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
										    self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
										    #~ cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
										    self.dev_closestLowerNoneNanIndex.data, self.dev_closestUpperNoneNanIndex.data, \
										    self.maxInterCoordinateAngle \
										    #~ self.dev_dbgOut.data, \
										    #~ self.dev_dbgOut2.data \
										    )

		barrierEvent = cl.enqueue_barrier(self.queue)
		
		#~ if self.getContourId() >= 14 and self.getNrOfTrackingIterations() >= 0:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_contourCenter.data,self.host_contourCenter).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#~ ipdb.set_trace()
		
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
		#~ if np.any(np.isnan(self.host_membraneCoordinatesX)) or np.any(np.isnan(self.host_membraneCoordinatesY)) or \
		   #~ np.any(np.isinf(self.host_membraneCoordinatesX)) or np.any(np.isinf(self.host_membraneCoordinatesY)) or \
		   #~ np.any(self.host_membraneCoordinatesX==0) or np.any(self.host_membraneCoordinatesY==0):
			#~ ipdb.set_trace()
			
		#~ if self.getContourId() == 70005 and self.getNrOfTrackingIterations() >= 3:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#~ ipdb.set_trace()
#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_previousInterpolatedMembraneCoordinatesX.data,self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_previousInterpolatedMembraneCoordinatesY.data,self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ plt.plot(self.host_previousInterpolatedMembraneCoordinatesX,self.host_previousInterpolatedMembraneCoordinatesY), plt.show()
			#~ ipdb.set_trace()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarRadius.data,self.host_membranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarTheta.data,self.host_membranePolarTheta).wait()
			#~ plt.plot(self.host_membranePolarRadius), plt.show()
			#~ plt.plot(self.host_membranePolarTheta), plt.show()
			#~ ipdb.set_trace()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembranePolarRadius.data,self.host_interpolatedMembranePolarRadius).wait()
			#~ plt.plot(self.host_interpolatedMembranePolarRadius), plt.show()
			#~ ipdb.set_trace()


		#if self.getContourId() == 2447 and self.getNrOfTrackingIterations() >= 19:
			##~ self.host_interpolationAnglesOld = np.float64(np.zeros(2048))
			##~ cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interpolationAnglesOld).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			##~ plt.plot(self.host_interpolationAnglesOld), plt.show()
			#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#cl.enqueue_read_buffer(self.queue,self.dev_listOfGoodCoordinates.data,self.host_listOfGoodCoordinates).wait()
			#plt.plot(self.host_listOfGoodCoordinates), plt.ylim([-0.1,1.1]), plt.show()
			##~ ipdb.set_trace()
		
		#if self.getContourId() == 2447 and self.getNrOfTrackingIterations() == 20:
			#cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interCoordinateAngles).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#plt.plot(self.host_interCoordinateAngles), plt.show()
			#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#ipdb.set_trace()

		#if self.getContourId() == 2448:
			#cl.enqueue_read_buffer(self.queue,self.dev_interCoordinateAngles.data,self.host_interCoordinateAngles).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#plt.plot(self.host_interCoordinateAngles), plt.show()
			#plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
			#ipdb.set_trace()
		
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
		#if self.computeDeviceId is 10:
			##~ print bla
			#self.prg.calculateContourCenter(self.queue, self.gradientGlobalSize, self.gradientGlobalSize, \
										   #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
										   #cl.LocalMemory(self.membraneNormalVectors_memSize), cl.LocalMemory(self.membraneNormalVectors_memSize), \
										   #self.dev_contourCenter.data \
										  #)
		
		#### Use this for GPU and when number of detected points >500
		#### NOTE: There is a in the OpenCL driver for the Intel CPU. So that in the funciton below,
		#### 	  the CLK_GLOBAL_MEM_FENCE is not respected correctly leading to incorrect results
		#if self.computeDeviceId is 10:
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

		#~ if self.getContourId() == 70005 and self.getNrOfTrackingIterations() >= 1:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarRadius.data,self.host_membranePolarRadius).wait()
			#~ plt.plot(self.host_membranePolarRadius), plt.show()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarTheta.data,self.host_membranePolarTheta).wait()
			#~ plt.plot(self.host_membranePolarTheta), plt.show()
#~ 
			#~ ipdb.set_trace()
		
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
		
		#~ if self.getContourId() >= 512 and self.getNrOfTrackingIterations() >= 1:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarRadius.data,self.host_membranePolarRadius).wait()
			#~ plt.plot(self.host_membranePolarRadius), plt.show()
			#~ 
			#~ membranePolarRadius_ref = np.copy(self.host_membranePolarRadius)
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarTheta.data,self.host_membranePolarTheta).wait()
			#~ plt.plot(self.host_membranePolarTheta), plt.show()
#~ 
			#~ membranePolarTheta_ref = np.copy(self.host_membranePolarTheta)
#~ 
			#~ ipdb.set_trace()

		#~ self.prg.filterPolarCoordinateSingularities_OLD(self.queue, self.gradientGlobalSize, None, \
													#~ self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
													#~ self.minAngleDifference
													#~ )
													
		#~ self.prg.calculateAngleDifference( self.queue, self.gradientGlobalSize, None, \
										   #~ self.dev_membranePolarTheta.data, \
										   #~ self.dev_angleDifferencesUpper.data, \
										   #~ self.dev_angleDifferencesLower.data \
										 #~ )

		#~ barrierEvent = cl.enqueue_barrier(self.queue)

		#~ self.prg.filterPolarCoordinateSingularities(self.queue, self.gradientGlobalSize, None, \
													#~ self.dev_membranePolarRadius.data, \
													#~ self.dev_membranePolarTheta.data, \
												    #~ self.dev_angleDifferencesUpper.data, \
												    #~ self.dev_angleDifferencesLower.data, \
													#~ self.dev_previousContourCenter.data, \
													#~ self.dev_membraneCoordinatesX.data, \
													#~ self.dev_membraneCoordinatesY.data, \
													#~ self.dev_membraneNormalVectorsX.data, \
													#~ self.dev_membraneNormalVectorsY.data, \
													#~ cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), \
													#~ cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
													#~ self.minAngleDifference \
										    		#~ )
													
													
		#self.prg.filterIncorrectCoordinates(self.queue, self.gradientGlobalSize, None, \
											#self.dev_previousContourCenter.data, \
										    #self.dev_interCoordinateAngles.data, \
										    #self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
										    #self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
										    #cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
										    #self.maxInterCoordinateAngle \
										    ##~ self.dev_dbgOut.data, \
										    ##~ self.dev_dbgOut2.data \
										    #)

		#~ barrierEvent = cl.enqueue_barrier(self.queue)

		#~ if self.getContourId() >= 512 and self.getNrOfTrackingIterations() >= 1:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarRadius.data,self.host_membranePolarRadius).wait()
			#~ plt.plot(self.host_membranePolarRadius), plt.show()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarTheta.data,self.host_membranePolarTheta).wait()
			#~ plt.plot(self.host_membranePolarTheta), plt.show()
#~ 
			#~ plt.plot(membranePolarTheta_ref,membranePolarRadius_ref,'r')
			#~ plt.plot(self.host_membranePolarTheta,self.host_membranePolarRadius)
			#~ plt.show()
#~ 
			#~ plt.plot(membranePolarRadius_ref)
			#~ plt.plot(self.host_membranePolarRadius,'r')
			#~ plt.show()
			#~ 
			#~ ipdb.set_trace()
		
		#~ self.prg.sortCoordinates(self.queue, (1,1), None, \
								#~ self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
								#~ self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								#~ self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
								#~ np.int32(self.nrOfDetectionAngleSteps) \
								#~ )
		
		#~ if self.getContourId() >= 83 and self.getNrOfTrackingIterations() >= 2:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarRadius.data,self.host_membranePolarRadius).wait()
			#~ plt.plot(self.host_membranePolarRadius), plt.show()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarTheta.data,self.host_membranePolarTheta).wait()
			#~ plt.plot(self.host_membranePolarTheta), plt.show()
#~ 
			#~ plt.plot(self.host_membranePolarTheta,self.host_membranePolarRadius)
			#~ plt.show()
#~ 
			#~ plt.plot(self.host_membranePolarRadius,'r')
			#~ plt.show()
			#~ 
			#~ ipdb.set_trace()

		#~ if self.getContourId() >= 83 and self.getNrOfTrackingIterations() >= 2:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut.data, self.host_dbgOut).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut2.data, self.host_dbgOut2).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
#~ 
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY)
			#~ plt.show()
#~ 
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

		
		#~ if self.getContourId() == 70005 and self.getNrOfTrackingIterations() >= 1:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_contourCenter.data,self.host_contourCenter).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesX.data,self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membraneCoordinatesY.data,self.host_membraneCoordinatesY).wait()
			#~ ind = 805
			#~ x = self.host_contourCenter['x'][0]
			#~ y = self.host_contourCenter['y'][0]
			#~ dx = self.host_membraneCoordinatesX[ind] - x
			#~ dy = self.host_membraneCoordinatesY[ind] - y
			#~ plt.arrow( x, y, dx, dy, fc="k" )
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY), plt.show()
#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembranePolarRadius.data,self.host_interpolatedMembranePolarRadius).wait()
			#~ plt.plot(self.host_interpolatedMembranePolarRadius), plt.show()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarRadius.data,self.host_membranePolarRadius).wait()
			#~ plt.plot(self.host_membranePolarRadius), plt.show()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_membranePolarTheta.data,self.host_membranePolarTheta).wait()
			#~ plt.plot(self.host_membranePolarTheta), plt.show()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolationAngles.data,self.host_interpolationAngles).wait()
			#~ plt.plot(self.host_interpolationAngles), plt.show()
			#~ 
			#~ ipdb.set_trace()


		#~ if self.getContourId() >= 11598 and self.getNrOfTrackingIterations() >= 14:
			#~ ipdb.set_trace()

		#~ self.prg.interpolatePolarCoordinates(self.queue, self.gradientGlobalSize, None, \
											#~ self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
											#~ self.dev_membranePolarRadiusTMP.data, \
											#~ self.dev_membranePolarThetaTMP.data, \
											#~ self.dev_membranePolarRadiusInterpolation.data, \
											#~ self.dev_membranePolarThetaInterpolation.data, \
											#~ self.dev_membranePolarRadiusInterpolationTesting.data, \
											#~ self.dev_membranePolarThetaInterpolationTesting.data, \
											#~ self.dev_interpolationAngles.data, \
											#~ self.nrOfInterpolationPoints, \
											#~ np.int32(self.nrOfDetectionAngleSteps), \
											#~ self.dev_interpolatedMembranePolarRadius.data, \
											#~ self.dev_b.data, self.dev_c.data, self.dev_d.data \
											#~ )
		
		self.prg.interpolatePolarCoordinatesLinear(self.queue, self.gradientGlobalSize, None, \
													self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
													self.dev_radialVectors.data, \
													self.dev_contourCenter.data, \
													self.dev_membraneCoordinatesX.data, \
													self.dev_membraneCoordinatesY.data, \
													self.dev_interpolatedMembraneCoordinatesX.data, \
													self.dev_interpolatedMembraneCoordinatesY.data, \
													self.dev_membranePolarRadiusTMP.data, \
													self.dev_membranePolarThetaTMP.data, \
													self.dev_membranePolarRadiusInterpolation.data, \
													self.dev_membranePolarThetaInterpolation.data, \
													self.dev_membranePolarRadiusInterpolationTesting.data, \
													self.dev_membranePolarThetaInterpolationTesting.data, \
													self.dev_interpolationAngles.data, \
													self.nrOfInterpolationPoints, \
													np.int32(self.nrOfDetectionAngleSteps), \
													self.nrOfAnglesToCompare, \
													self.dev_dbgOut.data, \
													self.dev_dbgOut2.data, \
													)
		#~ if self.getContourId() >= 14 and self.getNrOfTrackingIterations() >= 0:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY)
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY)
			#~ plt.show()
			#~ ipdb.set_trace()
		
		#~ if self.getContourId() >= 11598 and self.getNrOfTrackingIterations() >= 7:
			#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut.data, self.host_dbgOut).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut2.data, self.host_dbgOut2).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membranePolarTheta.data, self.host_membranePolarTheta).wait()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
			#~ 
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
#~ 
			#~ ipdb.set_trace()
#~ 
			#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY)
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY)
			#~ plt.show()
#~ 
			#~ plt.plot(self.host_interpolationAngles)
			#~ plt.plot(self.host_dbgOut)
			#~ plt.plot(self.host_dbgOut2)
			#~ plt.show()


		#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut.data, self.host_dbgOut).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_dbgOut2.data, self.host_dbgOut2).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolationAngles.data, self.host_interpolationAngles).wait()
		#~ 
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		#~ 
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		
		#~ ind=71
		#~ for ind in xrange(2048):
			#~ print "ind: "+str(ind)
			#~ print str(self.host_interpolationAngles[ind]>=self.host_dbgOut[ind] and self.host_interpolationAngles[ind]<self.host_dbgOut2[ind])
		#~ print str(np.all(self.host_interpolationAngles[:]>=self.host_dbgOut[:]) and np.all(self.host_interpolationAngles[:]<self.host_dbgOut2[:]))
		#~ ipdb.set_trace()
		
		#~ plt.plot(self.host_membraneCoordinatesX,self.host_membraneCoordinatesY)
		#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY)
		#~ plt.show()
		
		#~ plt.plot(self.host_interpolationAngles)
		#~ plt.plot(self.host_dbgOut)
		#~ plt.plot(self.host_dbgOut2)
		#~ plt.show()
		
		#~ if self.getContourId() >= 11598 and self.getNrOfTrackingIterations() >= 14:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembranePolarRadius.data,self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolationAngles.data,self.host_interpolationAngles).wait()
#~ 
			#~ plt.plot(self.host_membranePolarTheta,self.host_membranePolarRadius,'g')
			#~ plt.plot(self.host_interpolationAngles,self.host_interpolatedMembranePolarRadius,'b')
			#~ plt.show()
#~ 
			#~ plt.plot(self.host_membranePolarRadius,'g')
			#~ plt.plot(self.host_interpolatedMembranePolarRadius,'b')
			#~ plt.show()
			#~ ipdb.set_trace()


		#~ if self.getContourId() == 70005 and self.getNrOfTrackingIterations() >= 1:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembranePolarRadius.data,self.host_interpolatedMembranePolarRadius).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolationAngles.data,self.host_interpolationAngles).wait()
#~ 
			#~ plt.plot(membranePolarTheta_ref,membranePolarRadius_ref,'r')
			#~ plt.plot(self.host_membranePolarTheta,self.host_membranePolarRadius,'g')
			#~ plt.plot(self.host_interpolationAngles,self.host_interpolatedMembranePolarRadius,'b')
			#~ plt.show()
#~ 
			#~ plt.plot(membranePolarRadius_ref,'r')
			#~ plt.plot(self.host_membranePolarRadius,'g')
			#~ plt.plot(self.host_interpolatedMembranePolarRadius,'b')
			#~ plt.show()
			#~ ipdb.set_trace()

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
		
		#~ self.prg.pol2cart(self.queue, self.gradientGlobalSize, None, \
						  #~ self.dev_interpolatedMembraneCoordinatesX.data, self.dev_interpolatedMembraneCoordinatesY.data, \
						  #~ self.dev_interpolatedMembranePolarRadius.data, self.dev_interpolationAngles.data, \
						  #~ self.dev_contourCenter.data  )
#~ 
		#~ barrierEvent = cl.enqueue_barrier(self.queue)
#~ 
		#~ self.prg.calculateInterCoordinateAngles(self.queue, self.gradientGlobalSize, None, \
												#~ self.dev_interCoordinateAngles.data, \
												#~ self.dev_interpolatedMembraneCoordinatesX.data, self.dev_interpolatedMembraneCoordinatesY.data \
											   #~ )
		#~ 
		#~ barrierEvent = cl.enqueue_barrier(self.queue)
#~ 
		#~ self.prg.filterIncorrectCoordinates(self.queue, self.gradientGlobalSize, None, \
											#~ self.dev_previousContourCenter.data, \
										    #~ self.dev_interCoordinateAngles.data, \
										    #~ self.dev_interpolatedMembraneCoordinatesX.data, self.dev_interpolatedMembraneCoordinatesY.data, \
										    #~ self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
										    #~ cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
										    #~ self.maxInterCoordinateAngle \
										    #~ )
#~ 
		#~ barrierEvent = cl.enqueue_barrier(self.queue)
		
		#~ if self.getContourId() == 70005 and self.getNrOfTrackingIterations() >= 1:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesX.data,self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesY.data,self.host_interpolatedMembraneCoordinatesY).wait()
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY), plt.show()
#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_previousInterpolatedMembraneCoordinatesX.data,self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_previousInterpolatedMembraneCoordinatesY.data,self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ plt.plot(self.host_previousInterpolatedMembraneCoordinatesX,self.host_previousInterpolatedMembraneCoordinatesY), plt.show()
			#~ ipdb.set_trace()

		barrierEvent = cl.enqueue_barrier(self.queue)
			
		#~ self.prg.filterJumpedCoordinates(self.queue, self.gradientGlobalSize, None, \
											#~ self.dev_previousContourCenter.data, \
											#~ self.dev_interpolatedMembraneCoordinatesX.data, \
											#~ self.dev_interpolatedMembraneCoordinatesY.data, \
											#~ self.dev_membraneNormalVectorsX.data, \
											#~ self.dev_membraneNormalVectorsY.data, \
										    #~ self.dev_previousInterpolatedMembraneCoordinatesX.data, \
											#~ self.dev_previousInterpolatedMembraneCoordinatesY.data, \
										    #~ cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), \
											#~ cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
											#~ cl.LocalMemory(self.listOfGoodCoordinates_memSize), \
											#~ self.maxCoordinateShift, \
											#~ self.dev_listOfGoodCoordinates.data \
											#~ )
		
		#~ if self.getContourId() == 70005 and self.getNrOfTrackingIterations() >= 1:
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesX.data,self.host_interpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesY.data,self.host_interpolatedMembraneCoordinatesY).wait()
			#~ plt.plot(self.host_interpolatedMembraneCoordinatesX,self.host_interpolatedMembraneCoordinatesY), plt.show()
#~ 
			#~ cl.enqueue_read_buffer(self.queue,self.dev_previousInterpolatedMembraneCoordinatesX.data,self.host_previousInterpolatedMembraneCoordinatesX).wait()
			#~ cl.enqueue_read_buffer(self.queue,self.dev_previousInterpolatedMembraneCoordinatesY.data,self.host_previousInterpolatedMembraneCoordinatesY).wait()
			#~ plt.plot(self.host_previousInterpolatedMembraneCoordinatesX,self.host_previousInterpolatedMembraneCoordinatesY), plt.show()
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
		
		#~ self.dev_trackingFinished = cl_array.to_device(self.queue, self.host_trackingFinished)
		#~ self.dev_iterationFinished = cl_array.to_device(self.queue, self.host_iterationFinished)
		
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
		return self.host_interpolatedMembraneCoordinatesX/self.scalingFactor
		pass
	
	def getMembraneCoordinatesY(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		#~ return self.host_membraneCoordinatesY
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		return self.host_interpolatedMembraneCoordinatesY/self.scalingFactor
		pass

	def getMembraneCoordinatesXscaled(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesX.data, self.host_membraneCoordinatesX).wait()
		#~ return self.host_membraneCoordinatesX
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		return self.host_interpolatedMembraneCoordinatesX
		pass
	
	def getMembraneCoordinatesYscaled(self):
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
		self.host_contourCenter[0]['x']=self.host_contourCenter[0]['x']/self.scalingFactor
		self.host_contourCenter[0]['y']=self.host_contourCenter[0]['y']/self.scalingFactor
		return self.host_contourCenter
		pass

	def getFitInclines(self):
		#~ cl.enqueue_read_buffer(self.queue, self.dev_membraneCoordinatesY.data, self.host_membraneCoordinatesY).wait()
		#~ return self.host_membraneCoordinatesY
		#~ cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		cl.enqueue_read_buffer(self.queue, self.dev_fitInclines.data, self.host_fitInclines).wait()
		return self.host_fitInclines * self.scalingFactor # needs to be multiplied, since putting in more pixels artificially reduces the value of the incline
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
		snrRoiStartIndexes = self.snrRoi[0]
		snrRoiStopIndexes = self.snrRoi[1]
		return self.host_Img[snrRoiStartIndexes[0]:snrRoiStopIndexes[1],snrRoiStartIndexes[0]:snrRoiStopIndexes[1]]
