import numpy as np
import json
import PIL.Image as Image
from scipy import signal

class imagePreprocessor(object):
	"""description of class"""
	
	def __init__(self, configReader):
		self.configReader = configReader
		self.setup()
		#self.loadConfig(config)

	def setup(self):
		if self.configReader.performImageFiltering:
			self.setupFilter()

		self.scalingMethodVar = self.setImageScalingMethod(self.configReader.scalingMethod)

	
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
		
		if self.configReader.performImageFiltering is True:
			self.filterImage()
		
		if self.configReader.performImageScaling is True:
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
	
	def setupWienerFilter(self):
		self.imageFilter = self.wienerFilterMod
		self.filterArguments = (self.configReader.filterKernelSize,self.configReader.noisePowerEstimate)
		pass
		
	def wienerFilterMod(self,imageData):
		kernelSize = self.filterArguments[0]
		if self.filterArguments[1] == "estimateFromSnrRoi":
			noisePower = self.getImageStd()**2
		else:
			noisePower = self.filterArguments[1]
		filterArgs = (kernelSize,noisePower)
		return signal.wiener(imageData,*filterArgs)
		
	def setupFilter(self):
		if self.configReader.filterType == "wiener":
			self.setupWienerFilter()
		pass

	def getImageStd(self):
		roiValues = self.getRoiIntensityValues()
		return roiValues.std()

	def getRoiIntensityValues(self):
		snrRoi = self.getSnrRoi()
		snrRoiStartIndexes = snrRoi[0]
		snrRoiStopIndexes = snrRoi[1]
		return self.host_ImgUnfilteredUnscaled[snrRoiStartIndexes[0]:snrRoiStopIndexes[1],snrRoiStartIndexes[0]:snrRoiStopIndexes[1]]		
		
	def getSnrRoi(self):
		return self.configReader.snrRoi
		pass

	def getSnrRoiScaled(self):
		return np.floor(self.configReader.snrRoi*self.configReader.scalingFactor)
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
