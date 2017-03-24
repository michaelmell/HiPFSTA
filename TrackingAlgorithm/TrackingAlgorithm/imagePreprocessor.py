import numpy as np
import json
import PIL.Image as Image
from scipy import signal

class imagePreprocessor(object):
	"""description of class"""
	
	def __init__(self, config):
		self.loadConfig(config)

	def processImage(self, im):
		#im = Image.open(imagePath)
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
		
		return self.host_Img
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
		
	def getImageStd(self):
		roiValues = self.getRoiIntensityValues()
		return roiValues.std()

	def getRoiIntensityValues(self):
		snrRoi = self.getSnrRoi()
		snrRoiStartIndexes = snrRoi[0]
		snrRoiStopIndexes = snrRoi[1]
		return self.host_ImgUnfilteredUnscaled[snrRoiStartIndexes[0]:snrRoiStopIndexes[1],snrRoiStartIndexes[0]:snrRoiStopIndexes[1]]		
		
	def getSnrRoi(self):
		return self.snrRoi
		pass

	def getSnrRoiScaled(self):
		return np.floor(self.snrRoi*self.scalingFactor)
		pass

	def getImageSnr(self):
		roiStd = self.getImageStd()
		roiMean = self.getImageIntensity()
		return roiMean/roiStd
		pass

	def getImageIntensity(self):
		roiValues = self.getRoiIntensityValues()
		return roiValues.mean()
		pass
