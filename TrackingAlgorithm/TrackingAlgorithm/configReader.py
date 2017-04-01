import os
import configparser
import json
import numpy as np

class configReader(object):
	"""description of class"""
	def __init__(self,configurationFile,runInteractive=False):
		self.runInteractive = runInteractive
		self.readConfigFile(configurationFile)
		self.loadContourTrackerMainConfig()
		self.loadImagePreprocessingConfig()
		self.loadContourTrackerConfig()
		self.runConfigChecks()

	def loadContourTrackerConfig(self):
		self.startingCoordinate = self.scalingFactor * np.array(json.loads(self.config.get("TrackingParameters","startingCoordinate")))
		self.rotationCenterCoordinate = self.scalingFactor * np.array(json.loads(self.config.get("TrackingParameters","rotationCenterCoordinate")))
		
		self.linFitParameter = self.scalingFactor * np.float64(json.loads(self.config.get("TrackingParameters","linFitParameter")))
		self.linFitSearchRange = self.scalingFactor * np.float64(json.loads(self.config.get("TrackingParameters","linFitSearchRange")))
		self.interpolationFactor = np.int32(np.float64(json.loads(self.config.get("TrackingParameters","interpolationFactor")))/self.scalingFactor)
		
		self.meanParameter = np.int32(np.round(self.scalingFactor * np.float64(json.loads(self.config.get("TrackingParameters","meanParameter")))))
		self.meanRangePositionOffset = self.scalingFactor * np.float64(json.loads(self.config.get("TrackingParameters","meanRangePositionOffset")))

		self.localAngleRange = np.float64(json.loads(self.config.get("TrackingParameters","localAngleRange")))
		self.nrOfLocalAngleSteps = np.int32(json.loads(self.config.get("TrackingParameters","nrOfLocalAngleSteps")))
		
		self.detectionKernelStrideSize = np.int32(json.loads(self.config.get("TrackingParameters","detectionKernelStrideSize")))
		self.nrOfStrides = np.int32(json.loads(self.config.get("TrackingParameters","nrOfStrides")))
		
		self.nrOfAnglesToCompare = np.int32(json.loads(self.config.get("TrackingParameters","nrOfAnglesToCompare")))
		
		self.nrOfIterationsPerContour = np.int32(json.loads(self.config.get("TrackingParameters","nrOfIterationsPerContour")))
		
		self.computeDeviceId = json.loads(self.config.get("OpenClParameters","computeDeviceId"))

		self.inclineTolerance = np.float64(json.loads(self.config.get("TrackingParameters","inclineTolerance")))
		
		self.coordinateTolerance = self.scalingFactor * np.float64(json.loads(self.config.get("TrackingParameters","coordinateTolerance")))
		
		self.maxNrOfTrackingIterations = json.loads(self.config.get("TrackingParameters","maxNrOfTrackingIterations"))
		self.minNrOfTrackingIterations = json.loads(self.config.get("TrackingParameters","minNrOfTrackingIterations"))
		
		self.centerTolerance = self.scalingFactor * np.float64(json.loads(self.config.get("TrackingParameters","centerTolerance")))
		
		self.maxInterCoordinateAngle = np.float64(json.loads(self.config.get("TrackingParameters","maxInterCoordinateAngle")))
		self.maxCoordinateShift = self.scalingFactor * np.float64(json.loads(self.config.get("TrackingParameters","maxCoordinateShift")))
		
		resetNormalsAfterEachImage = self.config.get("TrackingParameters","resetNormalsAfterEachImage")
		if resetNormalsAfterEachImage == 'True':
			self.resetNormalsAfterEachImage = True
		else:
			self.resetNormalsAfterEachImage = False


	def loadImagePreprocessingConfig(self):
		self.filterKernelSize = json.loads(self.config.get("ImageFilterParameters","filterKernelSize"))
		if self.filterKernelSize == "None":
			self.filterKernelSize = None
		else:
			self.filterKernelSize = json.loads(self.config.get("ImageFilterParameters","filterKernelSize"))
		
		self.noisePowerEstimate = self.config.get("ImageFilterParameters","noisePowerEstimate")
		if self.noisePowerEstimate == "None":
			self.noisePowerEstimate = None
		else:
			self.noisePowerEstimate = json.loads(self.config.get("ImageFilterParameters","noisePowerEstimate"))

		self.filterType = json.loads(self.config.get("ImageFilterParameters","filterType"))

	# from: http://stackoverflow.com/questions/335695/lists-in-configparser
	# json.loads(self.config.get("SectionOne","startingCoordinate"))
		snrRoi = self.config.get("TrackingParameters","snrRoi")
		if snrRoi == "" or snrRoi == "None":
			self.snrRoi = None
		else:
			self.snrRoi = np.array(json.loads(self.config.get("TrackingParameters","snrRoi")))
		
		performImageFiltering = self.config.get("ImageFilterParameters","performImageFiltering")
		if performImageFiltering == "True":
			self.performImageFiltering = True
		else:
			self.performImageFiltering = False
		
		performImageScaling = self.config.get("ImageManipulationParameters","performImageScaling")

		if performImageScaling == "True":
			self.performImageScaling = True
		else:
			self.performImageScaling = False
		
		if self.performImageScaling == True:
			self.scalingFactor = np.float64(json.loads(self.config.get("ImageManipulationParameters","scalingFactor")))
		else:
			self.scalingFactor = np.float64(1)

		self.scalingMethod = json.loads(self.config.get("ImageManipulationParameters","scalingMethod"))

	def loadContourTrackerMainConfig(self):
		backgroundDirectoryPath = self.config.get("FileParameters","backgroundDirectoryPath")
		if backgroundDirectoryPath == "" or backgroundDirectoryPath == "None":
			self.backgroundDirectoryPath = None
		else:
			self.backgroundDirectoryPath = json.loads(backgroundDirectoryPath)
		
		darkfieldDirectoryPath = self.config.get("FileParameters","darkfieldDirectoryPath")
		if darkfieldDirectoryPath == "" or darkfieldDirectoryPath == "None":
			self.darkfieldDirectoryPath = None
		else:
			self.darkfieldDirectoryPath = json.loads(darkfieldDirectoryPath)
		
		self.imageDirectoryPath = json.loads(self.config.get("FileParameters","imageDirectoryPath"))
		
		ignoredImageIndices = self.config.get("FileParameters","ignoredImageIndices")
		if ignoredImageIndices == "" or ignoredImageIndices == "None":
			self.ignoredImageIndices = None
		else:
			if not ":" in ignoredImageIndices:
				self.ignoredImageIndices = np.array(json.loads(self.config.get("FileParameters","ignoredImageIndices")))
			else:
				self.ignoredImageIndices = self.parseIndexRanges(ignoredImageIndices)

			
		nrOfFramesToSaveFitInclinesFor = self.config.get("FileParameters","nrOfFramesToSaveFitInclinesFor")
		if nrOfFramesToSaveFitInclinesFor == "" or nrOfFramesToSaveFitInclinesFor == "None":
			self.nrOfFramesToSaveFitInclinesFor = None
		else:
			self.nrOfFramesToSaveFitInclinesFor = json.loads(self.config.get("FileParameters","nrOfFramesToSaveFitInclinesFor"))
		
		self.imageFileExtension = json.loads(self.config.get("FileParameters","imageFileExtension"))
		self.dataAnalysisDirectoryPath = json.loads(self.config.get("FileParameters","dataAnalysisDirectoryPath"))

		self.detectionKernelStrideSize = np.int32(json.loads(self.config.get("TrackingParameters","detectionKernelStrideSize")))
		self.nrOfStrides = np.int32(json.loads(self.config.get("TrackingParameters","nrOfStrides")))
		self.nrOfContourPoints = self.detectionKernelStrideSize*self.nrOfStrides
		self.nrOfDetectionAngleSteps = int(self.detectionKernelStrideSize * self.nrOfStrides)
		
		self.clPlatform = json.loads(self.config.get("OpenClParameters","clPlatform"))
		self.computeDeviceId = json.loads(self.config.get("OpenClParameters","computeDeviceId"))
		self.nrOfTrackingQueues = json.loads(self.config.get("OpenClParameters","nrOfTrackingQueues"))
		
		self.detectionKernelStrideSize = np.int32(json.loads(self.config.get("TrackingParameters","detectionKernelStrideSize")))
		self.nrOfStrides = np.int32(json.loads(self.config.get("TrackingParameters","nrOfStrides")))
		
		self.imageIndexToContinueFrom = json.loads(self.config.get("TrackingParameters","imageIndexToContinueFrom"))
		
		self.stepsBetweenSavingResults = json.loads(self.config.get("TrackingParameters","stepsBetweenSavingResults"))
		
		snrRoi = self.config.get("TrackingParameters","snrRoi")
		if snrRoi == "" or snrRoi == "None":
			self.snrRoi = None
		else:
			self.snrRoi = np.array(json.loads(self.config.get("TrackingParameters","snrRoi")))
		pass
		
	def readConfigFile(self,configurationFile):
		if os.path.isfile(configurationFile) is False:
			print("")
			print("\tError: Configuration file not found at: "+configurationFile)
			sys.exit(1)
				
		self.config = configparser.ConfigParser()
		self.config.read(configurationFile,encoding="utf8")

	def parseIndexRanges(self,ignoredImageIndices):
		ignoredImageIndicesTmp = ignoredImageIndices
		ignoredImageIndicesTmp = ignoredImageIndicesTmp.strip('[[')
		ignoredImageIndicesTmp = ignoredImageIndicesTmp.strip(']]')
		indexRanges = ignoredImageIndicesTmp.split(',')
		parsedIndexes = np.array((),dtype=np.int64)
		for indexRange in indexRanges:
			indexRangeTmp = indexRange.strip('[')
			indexRangeTmp = indexRangeTmp.strip(']')
			if ':' not in indexRangeTmp:
				parsedIndexes = np.append(parsedIndexes,np.int64(indexRangeTmp))
			else:
				[startIndex,endIndex] = indexRangeTmp.split(':')
				parsedIndexes = np.append(parsedIndexes,np.arange(np.int64(startIndex),np.int64(endIndex)+1))
		return parsedIndexes
		pass

	def runConfigChecks(self):
		if self.darkfieldDirectoryPath is not None:
			if not os.path.isdir(self.darkfieldDirectoryPath):
				print("")
				print("\tERROR: Directory at 'darkfieldDirectoryPath' does not exist.")
				sys.exit(1)
		if self.backgroundDirectoryPath is not None:
			if not os.path.isdir(self.backgroundDirectoryPath):
				print("")
				print("\tERROR: Directory at 'backgroundDirectoryPath' does not exist.")
				sys.exit(1)
		if not os.path.isdir(self.imageDirectoryPath):
			print("")
			print("\tERROR: Directory at 'imageDirectoryPath' does not exist.")
			sys.exit(1)
		if not os.path.isdir(self.dataAnalysisDirectoryPath):
			if self.runInteractive: # if we are not running interactive, we know what we're doing (e.g. overwriting by default)
				print("")
				print("\tWARNING: Directory at 'dataAnalysisDirectoryPath' does not exist. Create it?")
				print("")
				print("\t'dataAnalysisDirectoryPath':")
				print("\t"+self.dataAnalysisDirectoryPath)
				print("")
				answer = input("\tContinue? (y: yes, n: no) ")
				if answer.lower().startswith("y"):
					os.makedirs(self.dataAnalysisDirectoryPath)
				else:
					exit()
			else:
				print("")
				print("\tWARNING: Directory at 'dataAnalysisDirectoryPath' did not exist. Created it.")
				print("")
				os.makedirs(self.dataAnalysisDirectoryPath)

		if os.listdir(self.dataAnalysisDirectoryPath) != [] and self.imageIndexToContinueFrom == 0:
			if self.runInteractive: # if we are not running interactive, we know what we're doing (e.g. overwriting by default)
				print("")
				print("\tWARNING: Directory at 'dataAnalysisDirectoryPath' is not empty and 'imageIndexToContinueFrom' is 0. Continuing may result in data-loss.")
				print("")
				print("\t'dataAnalysisDirectoryPath':")
				print("\t"+self.dataAnalysisDirectoryPath)
				print("")
				answer = input("\tContinue? (y: yes, n: no) ")
				if answer.lower().startswith("n"):
					exit()
	