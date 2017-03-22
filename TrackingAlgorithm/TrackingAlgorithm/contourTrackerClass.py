import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
#~ import cv2 # OpenCV 2.3.1
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as Image
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
		self.setContourId(-1) # initialize the contour id to -1; this will later change at run time
		self.nrOfTrackingIterations = 0
		pass
	
	def setupClQueue(self,ctx):
		self.ctx = ctx
		self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
		self.mf = cl.mem_flags
		self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)
		pass
		
	def loadDarkfield(self, darkfieldList):
		if darkfieldList == []:
			self.darkfieldData = None
		else:
			for darkfieldIndex, darkfieldFile in enumerate(darkfieldList):
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
		imageOrig = self.host_Img
		imageNew = self.imageFilter(imageOrig)
		self.host_Img = np.array(imageNew,dtype=np.float32)
		pass
		
	def rescaleImage(self):
		# get image data into image structure for manipulations
		imgShape = np.asarray(self.host_ImgUnfilteredUnscaled.shape, dtype=np.float32)
		im = Image.fromarray(self.host_Img)
		imgShapeTmp = np.asarray([imgShape[1],imgShape[0]], dtype=np.float32)
		newImageShape = tuple(np.int32(np.round(self.scalingFactor * imgShapeTmp)))
		newImage = im.resize(newImageShape, self.scalingMethodVar)

		# read back manipulated image data to host image array
		imgDataTmp = list(newImage.getdata())
		self.host_Img = np.asarray(imgDataTmp, dtype=np.float32).reshape((newImageShape[1],newImageShape[0]))
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
		filterKernelSize = json.loads(config.get("ImageFilterParameters","filterKernelSize"))
		if filterKernelSize == "None":
			filterKernelSize = None
		else:
			filterKernelSize = json.loads(config.get("ImageFilterParameters","filterKernelSize"))
		
		noisePowerEstimate = config.get("ImageFilterParameters","noisePowerEstimate")
		if noisePowerEstimate == "None":
			noisePowerEstimate = None
			#~ if self.snrRoi is not None:
				#~ noisePowerEstimate = "estimateFromSnrRoi"
		else:
			noisePowerEstimate = json.loads(config.get("ImageFilterParameters","noisePowerEstimate"))
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
		
		self.linFitParameter = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","linFitParameter")))
		self.linFitSearchRange = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","linFitSearchRange")))
		self.interpolationFactor = np.int32(np.float64(json.loads(config.get("TrackingParameters","interpolationFactor")))/self.scalingFactor)
		
		self.meanParameter = np.int32(np.round(self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","meanParameter")))))
		self.meanRangePositionOffset = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","meanRangePositionOffset")))

		self.localAngleRange = np.float64(json.loads(config.get("TrackingParameters","localAngleRange")))
		self.nrOfLocalAngleSteps = np.int32(json.loads(config.get("TrackingParameters","nrOfLocalAngleSteps")))
		
		self.detectionKernelStrideSize = np.int32(json.loads(config.get("TrackingParameters","detectionKernelStrideSize")))
		self.nrOfStrides = np.int32(json.loads(config.get("TrackingParameters","nrOfStrides")))
		
		self.nrOfAnglesToCompare = np.int32(json.loads(config.get("TrackingParameters","nrOfAnglesToCompare")))
		
		self.nrOfIterationsPerContour = np.int32(json.loads(config.get("TrackingParameters","nrOfIterationsPerContour")))
		
		self.computeDeviceId = json.loads(config.get("OpenClParameters","computeDeviceId"))
		
		self.coordinateTolerance = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","coordinateTolerance")))
		
		self.maxNrOfTrackingIterations = json.loads(config.get("TrackingParameters","maxNrOfTrackingIterations"))
		self.minNrOfTrackingIterations = json.loads(config.get("TrackingParameters","minNrOfTrackingIterations"))
		
		self.inclineTolerance = np.float64(json.loads(config.get("TrackingParameters","inclineTolerance")))
		
		self.centerTolerance = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","centerTolerance")))
		
		self.maxInterCoordinateAngle = np.float64(json.loads(config.get("TrackingParameters","maxInterCoordinateAngle")))
		self.maxCoordinateShift = self.scalingFactor * np.float64(json.loads(config.get("TrackingParameters","maxCoordinateShift")))
		
		resetNormalsAfterEachImage = config.get("TrackingParameters","resetNormalsAfterEachImage")
		if resetNormalsAfterEachImage == 'True':
			self.resetNormalsAfterEachImage = True
		else:
			self.resetNormalsAfterEachImage = False
		
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
		for index in range(localAngles.shape[0]):
			localAngle = localAngles[index]
			localRotationMatrices[index,:,:] = np.array([[np.cos(localAngle),-np.sin(localAngle)],[np.sin(localAngle),np.cos(localAngle)]])

		self.buf_localRotationMatrices = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=localRotationMatrices)

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
		
		self.host_localMembranePositionsX = np.zeros(shape=self.nrOfLocalAngleSteps,dtype=np.float64)
		self.dev_localMembranePositionsX = cl_array.to_device(self.queue, self.host_localMembranePositionsX)

		self.host_localMembranePositionsY = np.zeros(shape=self.nrOfLocalAngleSteps,dtype=np.float64)
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
		self.localMembranePositions_memSize = self.dev_localMembranePositionsX.nbytes * int(self.contourPointsPerWorkGroup)
		self.fitIncline_memSize = self.dev_fitIncline.nbytes * int(self.contourPointsPerWorkGroup)
		self.fitIntercept_memSize = self.dev_fitIntercept.nbytes * int(self.contourPointsPerWorkGroup)
		self.fitIntercept_memSize = self.dev_fitIntercept.nbytes * int(self.contourPointsPerWorkGroup)
		
		self.membranePolarTheta_memSize = self.dev_membranePolarTheta.nbytes
		self.membranePolarRadius_memSize = self.dev_membranePolarRadius.nbytes
		
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
		barrierEvent = cl.enqueue_barrier(self.queue)
		
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

	def trackContour(self):
		# tracking status variables
		self.nrOfTrackingIterations = self.nrOfTrackingIterations + 1
		
		stopInd = 1

		self.trackingFinished = np.array(1,dtype=np.int32) # True
		self.dev_trackingFinished = cl_array.to_device(self.queue, self.trackingFinished)
		
		self.iterationFinished = np.array(0,dtype=np.int32) # True
		self.dev_iterationFinished = cl_array.to_device(self.queue, self.iterationFinished)
		
		for strideNr in range(self.nrOfStrides):
			# set the starting index of the coordinate array for each kernel instance
			kernelCoordinateStartingIndex = np.int32(strideNr*self.detectionKernelStrideSize)
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
											 self.inclineTolerance)
			barrierEvent = cl.enqueue_barrier(self.queue)

		barrierEvent = cl.enqueue_barrier(self.queue)
		
		self.prg.filterNanValues(self.queue, self.gradientGlobalSize, None, \
								 self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								 self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
								 cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes) \
								 #~ self.dev_dbgOut.data, \
								 #~ self.dev_dbgOut2.data \
								 )
		barrierEvent = cl.enqueue_barrier(self.queue)

		barrierEvent = cl.enqueue_barrier(self.queue)
		
		self.prg.filterJumpedCoordinates(self.queue, self.gradientGlobalSize, None, \
											self.dev_previousContourCenter.data, \
											self.dev_membraneCoordinatesX.data, \
											self.dev_membraneCoordinatesY.data, \
											self.dev_membraneNormalVectorsX.data, \
											self.dev_membraneNormalVectorsY.data, \
										    self.dev_previousInterpolatedMembraneCoordinatesX.data, \
											self.dev_previousInterpolatedMembraneCoordinatesY.data, \
										    cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), \
											cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
											cl.LocalMemory(self.listOfGoodCoordinates_memSize), \
											self.maxCoordinateShift, \
											self.dev_listOfGoodCoordinates.data \
											)
		barrierEvent = cl.enqueue_barrier(self.queue)

		barrierEvent = cl.enqueue_barrier(self.queue)

		self.prg.calculateInterCoordinateAngles(self.queue, self.gradientGlobalSize, None, \
												self.dev_interCoordinateAngles.data, \
												self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data \
											   )
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.prg.filterIncorrectCoordinates(self.queue, self.gradientGlobalSize, None, \
											self.dev_previousContourCenter.data, \
										    self.dev_interCoordinateAngles.data, \
										    self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
										    self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
										    cl.LocalMemory(self.dev_closestLowerNoneNanIndex.nbytes), cl.LocalMemory(self.dev_closestUpperNoneNanIndex.nbytes), \
										    self.maxInterCoordinateAngle \
										    #~ self.dev_dbgOut.data, \
										    #~ self.dev_dbgOut2.data \
										    )
		barrierEvent = cl.enqueue_barrier(self.queue)
		
		# information regarding barriers: http://stackoverflow.com/questions/13200276/what-is-the-difference-between-clenqueuebarrier-and-clfinish

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
		
		self.prg.calculateDs(self.queue, self.gradientGlobalSize, None, \
					   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
					   self.dev_ds.data \
					 )
		barrierEvent = cl.enqueue_barrier(self.queue)

		barrierEvent = cl.enqueue_barrier(self.queue)

		self.prg.calculateSumDs(self.queue, self.gradientGlobalSize, None, \
					   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
					   self.dev_ds.data, self.dev_sumds.data \
					 )
		barrierEvent = cl.enqueue_barrier(self.queue)
		
		self.prg.calculateContourCenterNew2(self.queue, (1,1), None, \
								   self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								   self.dev_ds.data, self.dev_sumds.data, \
								   self.dev_contourCenter.data, \
								   np.int32(self.nrOfDetectionAngleSteps) \
								  )
		barrierEvent = cl.enqueue_barrier(self.queue)

		########################################################################
		### Convert cartesian coordinates to polar coordinates
		########################################################################
		self.prg.cart2pol(self.queue, self.gradientGlobalSize, None, \
						  self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
						  self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
						  self.dev_contourCenter.data)
		barrierEvent = cl.enqueue_barrier(self.queue)

		barrierEvent = cl.enqueue_barrier(self.queue)

		########################################################################
		### Interpolate polar coordinates
		########################################################################
		self.prg.sortCoordinates(self.queue, (1,1), None, \
								self.dev_membranePolarRadius.data, self.dev_membranePolarTheta.data, \
								self.dev_membraneCoordinatesX.data, self.dev_membraneCoordinatesY.data, \
								self.dev_membraneNormalVectorsX.data, self.dev_membraneNormalVectorsY.data, \
								np.int32(self.nrOfDetectionAngleSteps) \
								)
		barrierEvent = cl.enqueue_barrier(self.queue)

		barrierEvent = cl.enqueue_barrier(self.queue)

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
		barrierEvent = cl.enqueue_barrier(self.queue)
		
		barrierEvent = cl.enqueue_barrier(self.queue)

		########################################################################
		### Convert polar coordinates to cartesian coordinates
		########################################################################
		barrierEvent = cl.enqueue_barrier(self.queue)

		barrierEvent = cl.enqueue_barrier(self.queue)
		
		self.prg.checkIfTrackingFinished(self.queue, self.gradientGlobalSize, None, \
										 self.dev_interpolatedMembraneCoordinatesX.data, \
										 self.dev_interpolatedMembraneCoordinatesY.data, \
										 self.dev_previousInterpolatedMembraneCoordinatesX.data, \
										 self.dev_previousInterpolatedMembraneCoordinatesY.data, \
										 self.dev_trackingFinished.data, \
										 self.coordinateTolerance)
		barrierEvent = cl.enqueue_barrier(self.queue)

		self.prg.checkIfCenterConverged(self.queue, (1,1), None, \
										self.dev_contourCenter.data, \
										self.dev_previousContourCenter.data, \
										self.dev_trackingFinished.data, \
										self.centerTolerance)
		barrierEvent = cl.enqueue_barrier(self.queue)
		
		cl.enqueue_read_buffer(self.queue, self.dev_trackingFinished.data, self.trackingFinished).wait()
		barrierEvent = cl.enqueue_barrier(self.queue)

		cl.enqueue_copy_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesX.data,self.dev_previousInterpolatedMembraneCoordinatesX.data).wait()
		cl.enqueue_copy_buffer(self.queue,self.dev_interpolatedMembraneCoordinatesY.data,self.dev_previousInterpolatedMembraneCoordinatesY.data).wait()

		cl.enqueue_copy_buffer(self.queue,self.dev_contourCenter.data,self.dev_previousContourCenter.data).wait()

		# set variable to tell host program that the tracking iteration has finished
		self.prg.setIterationFinished(self.queue, (1,1), None, self.dev_iterationFinished.data)
		barrierEvent = cl.enqueue_barrier(self.queue)
		cl.enqueue_read_buffer(self.queue, self.dev_iterationFinished.data, self.iterationFinished).wait()
		pass
		
	def checkTrackingFinished(self):
		if self.nrOfTrackingIterations < self.minNrOfTrackingIterations:
			self.trackingFinished = 0 # force another iterations
		if self.nrOfTrackingIterations >= self.maxNrOfTrackingIterations:
			self.trackingFinished = 1 # force finish
		return self.trackingFinished
		pass
		
	def getMembraneCoordinatesX(self):
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesX.data, self.host_interpolatedMembraneCoordinatesX).wait()
		return self.host_interpolatedMembraneCoordinatesX/self.scalingFactor
		pass
	
	def getMembraneCoordinatesY(self):
		cl.enqueue_read_buffer(self.queue, self.dev_interpolatedMembraneCoordinatesY.data, self.host_interpolatedMembraneCoordinatesY).wait()
		return self.host_interpolatedMembraneCoordinatesY/self.scalingFactor
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
		self.host_contourCenter[0]['x']=self.host_contourCenter[0]['x']/self.scalingFactor
		self.host_contourCenter[0]['y']=self.host_contourCenter[0]['y']/self.scalingFactor
		return self.host_contourCenter
		pass

	def getFitInclines(self):
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
		return self.host_ImgUnfilteredUnscaled[snrRoiStartIndexes[0]:snrRoiStopIndexes[1],snrRoiStartIndexes[0]:snrRoiStopIndexes[1]]		

