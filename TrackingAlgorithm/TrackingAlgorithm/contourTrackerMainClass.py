from contourTrackerClass import contourTracker
from imagePreprocessor import imagePreprocessor

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import scipy.io as io
import glob
import ipdb
import os
import sys
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from configReader import configReader

class contourTrackerMain( object ):
	def __init__(self, configurationFile,runInteractive=False):
		self.runInteractive = runInteractive
		self.setInteractive(runInteractive)
		#~ self.configurationFile = configurationFile
		#self.loadSettings(configurationFile)
		self.configReader = configReader(configurationFile,runInteractive)
		self.getImageFileList()
		self.getDarkfieldFileList()
		self.getBackgroundFileList()
		self.imagePreprocessor = imagePreprocessor(self.configReader)
		self.imagePreprocessor.loadDarkfield(self.darkfieldList) # load darkfield images
		self.imagePreprocessor.loadBackground(self.backgroundList) # load background images
		self.setupClContext()
		self.setupManagementQueue()
		self.setupClVariables()
		self.printTrackingSetupInformation()
		self.setupTrackingQueues()
		
		if self.configReader.nrOfFramesToSaveFitInclinesFor:
			self.fitInclines = np.empty((self.configReader.nrOfContourPoints,self.configReader.nrOfFramesToSaveFitInclinesFor),dtype=np.float64)
		
		if self.configReader.snrRoi is not None:
			self.imageSnr = np.zeros((1,self.totalNrOfImages),dtype=np.float64)
			self.imageIntensity = np.zeros((1,self.totalNrOfImages),dtype=np.float64)
		
		self.contourCoordinatesX = np.empty((self.configReader.nrOfContourPoints,self.totalNrOfImages),dtype=np.float64)
		self.contourCoordinatesY = np.empty((self.configReader.nrOfContourPoints,self.totalNrOfImages),dtype=np.float64)
		
		self.contourNormalVectorsX = np.empty((self.configReader.nrOfContourPoints,self.totalNrOfImages),dtype=np.float64)
		self.contourNormalVectorsY = np.empty((self.configReader.nrOfContourPoints,self.totalNrOfImages),dtype=np.float64)
		
		self.contourCenterCoordinatesX = np.empty(self.totalNrOfImages,dtype=np.float64)
		self.contourCenterCoordinatesY = np.empty(self.totalNrOfImages,dtype=np.float64)
		
		self.nrOfIterationsPerContour = np.zeros(self.totalNrOfImages,dtype=np.int32)
		self.executionTimePerContour = np.zeros(self.totalNrOfImages,dtype=np.float64)
		
	def setProfiling(self,boolean):
		#~ self.profiling = boolean
		self.runInteractive = not boolean # using setProfiling with argument 'True' will deactivate interactivity
		pass

	def setInteractive(self,boolean):
		#~ self.profiling = boolean
		self.runInteractive = boolean # using setProfiling with argument 'True' will deactivate interactivity
		pass

	def startTracking(self):
		if self.configReader.imageIndexToContinueFrom is 0:
			self.doInitialTracking()
		else:
			self.setupContinuationOfTracking()
		self.__track()
		pass

	def setupClVariables(self):
		self.nrOfDetectionAngleSteps = self.configReader.nrOfDetectionAngleSteps
		self.host_mostRecentMembraneCoordinatesX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_mostRecentMembraneCoordinatesX = cl_array.to_device(self.managementQueue, self.host_mostRecentMembraneCoordinatesX)
		self.host_mostRecentMembraneCoordinatesY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_mostRecentMembraneCoordinatesY = cl_array.to_device(self.managementQueue, self.host_mostRecentMembraneCoordinatesY)
		
		self.host_mostRecentMembraneNormalVectorsX = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_mostRecentMembraneNormalVectorsX = cl_array.to_device(self.managementQueue, self.host_mostRecentMembraneNormalVectorsX)
		self.host_mostRecentMembraneNormalVectorsY = np.zeros(shape=self.nrOfDetectionAngleSteps,dtype=np.float64)
		self.dev_mostRecentMembraneNormalVectorsY = cl_array.to_device(self.managementQueue, self.host_mostRecentMembraneNormalVectorsY)

		self.host_contourCenter = np.zeros(1, cl.array.vec.double2)
		self.dev_mostRecentContourCenter = cl_array.to_device(self.managementQueue, self.host_contourCenter)
		pass
		
	def setupContinuationOfTracking(self):
		self.startTime = time.time()

		#~ self.currentImageIndex = self.imageIndexToContinueFrom
		#~ self.mostRecentImageIndex = self.imageIndexToContinueFrom-1
		#~ self.nrOfFinishedImages = self.imageIndexToContinueFrom-1
		#~ self.imageIndexToContinueFrom = self.imageIndexToContinueFrom-1
		
		self.imageIndexToContinueFrom = self.imageIndexToContinueFrom-1 # this shift in index is necessary so that we continue tracking at 'self.imageIndexToContinueFrom + 1' and not 'self.imageIndexToContinueFrom + 2'
		self.currentImageIndex = self.imageIndexToContinueFrom+1
		self.mostRecentImageIndex = self.imageIndexToContinueFrom
		self.nrOfFinishedImages = self.imageIndexToContinueFrom
		
		# load previous tracking data
		tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/contourCenterCoordinatesX.mat')
		self.contourCenterCoordinatesX = tmp['contourCenterCoordinatesX'][0]
		tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/contourCenterCoordinatesY.mat')
		self.contourCenterCoordinatesY = tmp['contourCenterCoordinatesY'][0]
		
		#~ ipdb.set_trace()
		
		tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/contourCoordinatesX.mat')
		self.contourCoordinatesX = tmp['contourCoordinatesX']
		tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/contourCoordinatesY.mat')
		self.contourCoordinatesY = tmp['contourCoordinatesY']
		
		if self.nrOfFramesToSaveFitInclinesFor:
			tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/fitInclines.mat')
			self.fitInclines = tmp['fitInclines']
		
		if self.snrRoi is not None:
			tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/imageSnr.mat')
			self.imageSnr = tmp['imageSnr']
			tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/imageIntensity.mat')
			self.imageIntensity = tmp['imageIntensity']

		tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/contourNormalVectorsX.mat')
		self.contourNormalVectorsX = tmp['contourNormalVectorsX']
		tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/contourNormalVectorsY.mat')
		self.contourNormalVectorsY = tmp['contourNormalVectorsY']
		
		self.host_membraneCoordinatesX = self.contourCoordinatesX[:,self.imageIndexToContinueFrom]
		self.host_membraneCoordinatesY = self.contourCoordinatesY[:,self.imageIndexToContinueFrom]

		self.host_membraneNormalVectorsX = self.contourNormalVectorsX[:,self.imageIndexToContinueFrom]
		self.host_membraneNormalVectorsY = self.contourNormalVectorsY[:,self.imageIndexToContinueFrom]
		
		# copy last tracked contour to GPU
		#~ self.dev_membraneCoordinatesX = cl_array.to_device(self.queue, self.host_membraneCoordinatesX)
		#~ self.dev_membraneCoordinatesY = cl_array.to_device(self.queue, self.host_membraneCoordinatesY)
		self.dev_mostRecentMembraneCoordinatesX = cl_array.to_device(self.managementQueue, self.host_membraneCoordinatesX)
		self.dev_mostRecentMembraneCoordinatesY = cl_array.to_device(self.managementQueue, self.host_membraneCoordinatesY)
		
		#~ cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneCoordinatesX.data,self.dev_mostRecentMembraneCoordinatesX.data).wait()
		#~ cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneCoordinatesY.data,self.dev_mostRecentMembraneCoordinatesY.data).wait()

		#~ cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneNormalVectorsX.data,self.dev_mostRecentMembraneNormalVectorsX.data).wait()
		#~ cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneNormalVectorsY.data,self.dev_mostRecentMembraneNormalVectorsY.data).wait()

		self.dev_mostRecentMembraneNormalVectorsX = cl_array.to_device(self.managementQueue, self.host_membraneNormalVectorsX)
		self.dev_mostRecentMembraneNormalVectorsY = cl_array.to_device(self.managementQueue, self.host_membraneNormalVectorsY)
		
		#~ ipdb.set_trace()
		
		pass

	def doInitialTracking(self):
		self.startTime = time.time()
		
		self.currentImageIndex = 0
		self.mostRecentImageIndex = 0
		self.nrOfFinishedImages = 0
		
		self.sequentialTracker.loadImage(self.imageList[self.currentImageIndex]) # load first image for initial tracking
		self.sequentialTracker.trackContourSequentially()
		
		if self.runInteractive:
			#~ plt.imshow(self.sequentialTracker.host_Img)
			plt.matshow(self.sequentialTracker.host_Img)
			plt.plot(self.sequentialTracker.getMembraneCoordinatesXscaled()-0.5,self.sequentialTracker.getMembraneCoordinatesYscaled()-0.5,'k')
			if self.snrRoi is not None:
				self.drawSnrRoiRectangle()
			self.printTrackingParameters()
			plt.show()
		
		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneCoordinatesX.data,self.dev_mostRecentMembraneCoordinatesX.data).wait()
		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneCoordinatesY.data,self.dev_mostRecentMembraneCoordinatesY.data).wait()

		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneNormalVectorsX.data,self.dev_mostRecentMembraneNormalVectorsX.data).wait()
		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneNormalVectorsY.data,self.dev_mostRecentMembraneNormalVectorsY.data).wait()

		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_contourCenter.data,self.dev_mostRecentContourCenter.data).wait()
		self.managementQueue.finish()
		
	def drawSnrRoiRectangle(self):
		snrRoi = self.imagePreprocessor.getSnrRoiScaled()
		snrRoiStartIndexes = snrRoi[0]
		snrRoiStopIndexes = snrRoi[1]
		plt.plot((snrRoiStartIndexes[0], snrRoiStartIndexes[0]),(snrRoiStartIndexes[1], snrRoiStopIndexes[1]),'k')
		plt.plot((snrRoiStopIndexes[0], snrRoiStopIndexes[0]),(snrRoiStartIndexes[1], snrRoiStopIndexes[1]),'k')
		plt.plot((snrRoiStartIndexes[0], snrRoiStopIndexes[0]),(snrRoiStopIndexes[1], snrRoiStopIndexes[1]),'k')
		plt.plot((snrRoiStartIndexes[0], snrRoiStopIndexes[0]),(snrRoiStartIndexes[1], snrRoiStartIndexes[1]),'k')
	
	def printTrackingParameters(self):
		print("")
		print("\tSome tracking parameters and image properties:")
		print("")
		self.printImageIntensityMsg()
		self.printImageStdMsg()
		self.printImageSnrMsg()
		print("\tUse image filtering:")
		print("\t"+str(self.sequentialTracker.performImageFiltering))
		print("")
		print("\tUse image scaling:")
		print("\t"+str(self.sequentialTracker.performImageScaling))
		print("")
		print("")
		
	def printImageIntensityMsg(self):
		print("\tImage intensity (obtained from 'snrRoi'):")
		if self.snrRoi is not None:
			print("\t"+str(self.sequentialTracker.getImageIntensity()))
		else:
			print("\tNo snrRoi provided.")
		print("")
		pass
	
	def printImageSnrMsg(self):
		print("\tImage SNR=intensity/std (obtained from 'snrRoi'):")
		if self.snrRoi is not None:
			print("\t"+str(self.sequentialTracker.getImageSnr()))
		else:
			print("\tNo snrRoi provided.")
		print("")
		pass
	
	def printImageStdMsg(self):
		print("\tImage STD (obtained from 'snrRoi'):")
		if self.snrRoi is not None:
			print("\t"+str(self.sequentialTracker.getImageStd()))
		else:
			print("\tNo snrRoi provided.")
		print("")
		pass
	
	def __track(self):
		# start tracking for all tracking-queues
		for tracker in self.trackingQueues:
			tracker.loadImage(self.imageList[self.currentImageIndex])
			tracker.setContourId(self.currentImageIndex)
			
			tracker.setStartingCoordinatesNew(self.dev_mostRecentMembraneCoordinatesX, \
											  self.dev_mostRecentMembraneCoordinatesY)
			tracker.setStartingMembraneNormals(self.dev_mostRecentMembraneNormalVectorsX, \
											   self.dev_mostRecentMembraneNormalVectorsY)
			
			cl.enqueue_copy_buffer(self.managementQueue,self.dev_mostRecentMembraneCoordinatesX.data,tracker.dev_previousInterpolatedMembraneCoordinatesX.data).wait()
			cl.enqueue_copy_buffer(self.managementQueue,self.dev_mostRecentMembraneCoordinatesY.data,tracker.dev_previousInterpolatedMembraneCoordinatesY.data).wait()

			cl.enqueue_copy_buffer(self.managementQueue,self.dev_mostRecentContourCenter.data,tracker.dev_previousContourCenter.data).wait()

			tracker.startTimer()
			tracker.trackContour()
			self.currentImageIndex = self.currentImageIndex + 1
		
		while(self.nrOfFinishedImages<self.totalNrOfImages): # enter control-loop for checking an controlling the states of the tracking-queues
			for tracker in self.trackingQueues:
				if tracker.iterationFinished:
					if tracker.checkTrackingFinished():
						self.nrOfFinishedImages = self.nrOfFinishedImages + 1
						
						# get tracking results
						self.writeContourToFinalArray(tracker)
						
						# update most recent contour profile
						if tracker.getContourId() >= self.mostRecentImageIndex:
							cl.enqueue_copy_buffer(self.managementQueue,tracker.dev_interpolatedMembraneCoordinatesX.data,self.dev_mostRecentMembraneCoordinatesX.data).wait()
							cl.enqueue_copy_buffer(self.managementQueue,tracker.dev_interpolatedMembraneCoordinatesY.data,self.dev_mostRecentMembraneCoordinatesY.data).wait()
							cl.enqueue_copy_buffer(self.managementQueue,tracker.dev_membraneNormalVectorsX.data,self.dev_mostRecentMembraneNormalVectorsX.data).wait()
							cl.enqueue_copy_buffer(self.managementQueue,tracker.dev_membraneNormalVectorsY.data,self.dev_mostRecentMembraneNormalVectorsY.data).wait()
							
							cl.enqueue_copy_buffer(self.managementQueue,tracker.dev_previousContourCenter.data,self.dev_mostRecentContourCenter.data).wait()

							self.mostRecentImageIndex = self.currentImageIndex
						
						frameId = tracker.getContourId()
						self.nrOfIterationsPerContour[frameId] = np.int32(tracker.getNrOfTrackingIterations())
						print("Nr of iterations: "+str(tracker.getNrOfTrackingIterations()))
						self.executionTimePerContour[frameId] = np.float64(tracker.getExectionTime())
						print("Execution time: "+str(self.executionTimePerContour[frameId])+" sec")
						
						self.currentTime = time.time()
						runningTime = self.currentTime - self.startTime
						print("Total running time: "+str(datetime.timedelta(seconds=runningTime))+" h")
						remainingTime = (self.totalNrOfImages-(self.currentImageIndex+1))*(runningTime/(self.currentImageIndex+1))
						print("Remaining running time (ETA): "+str(datetime.timedelta(seconds=remainingTime))+" h")
						print("\n")
						
						# do intermediate save points
						if self.mostRecentImageIndex % self.configReader.stepsBetweenSavingResults is 0:
							print("Saving intermediate results.")
							print("\n")
							self.saveTrackingResult()
						
						# start tracking of new image
						if self.currentImageIndex < self.totalNrOfImages:
							tracker.resetNrOfTrackingIterations()
							tracker.startTimer()
							
							print("Tracking image: "+str(self.currentImageIndex+1)+" of "+str(self.totalNrOfImages)) # 'self.currentImageIndex+1', because 'self.currentImageIndex' is zero-based index 
							print("Image File: "+os.path.basename(self.imageList[self.currentImageIndex])) # 'self.currentImageIndex+1', because 'self.currentImageIndex' is zero-based index 
							
							tracker.loadImage(self.imageList[self.currentImageIndex])
							tracker.setContourId(self.currentImageIndex)
							
							tracker.setStartingCoordinatesNew(self.dev_mostRecentMembraneCoordinatesX, \
															  self.dev_mostRecentMembraneCoordinatesY)
							tracker.setStartingMembraneNormals(self.dev_mostRecentMembraneNormalVectorsX, \
															   self.dev_mostRecentMembraneNormalVectorsY)
							
							tracker.trackContour()

							self.currentImageIndex = self.currentImageIndex + 1
						
					else: # start new tracking iteration with the previous contour as starting position
						tracker.setStartingCoordinatesNew(tracker.dev_interpolatedMembraneCoordinatesX, \
														  tracker.dev_interpolatedMembraneCoordinatesY \
													     )
						
						tracker.trackContour()

		print("Tracking finished. Saving results.")
		self.saveTrackingResult()

		self.currentTime = time.time()
		runningTime = self.currentTime - self.startTime
		print("Total running time: "+str(datetime.timedelta(seconds=runningTime))+" h")
		
		pass
		
	def setupClContext(self):
		self.clPlatformList = cl.get_platforms()
		counter = 0
		for platform in self.clPlatformList:
			if self.configReader.clPlatform in platform.name.lower():
				self.platformIndex = counter
			counter = counter + 1
		clDevicesList = self.clPlatformList[self.platformIndex].get_devices()
		
		computeDeviceIdSelection = self.configReader.computeDeviceId # 0: AMD-GPU; 1: Intel CPU
		self.device = clDevicesList[computeDeviceIdSelection]
		self.ctx = cl.Context([self.device])
		pass
		
	def setupManagementQueue(self):
		self.managementQueue = cl.CommandQueue(self.ctx)
		self.mf = cl.mem_flags
		pass

	def setupTrackingQueues(self):
		self.trackingQueues = [contourTracker(self.ctx, self.configReader, self.imagePreprocessor) for count in range(self.configReader.nrOfTrackingQueues)]
		self.sequentialTracker = contourTracker(self.ctx, self.configReader, self.imagePreprocessor)
		pass

	def getImageFileList(self):
		self.imageList = glob.glob(self.configReader.imageDirectoryPath+"/*."+self.configReader.imageFileExtension)
		self.imageList.sort() # or else their order is random...

		if self.configReader.ignoredImageIndices is not None:
			self.removeIgnoredImages()
			pass
		self.totalNrOfImages = self.imageList.__len__()
		pass
		
	def removeIgnoredImages(self):
		self.ignoredImageIndices.sort()
		for imageIndex in reversed(self.ignoredImageIndices): # remove in reverse order since we change the indexation, if we remove from small to large indexes
			del self.imageList[imageIndex-1]
		pass
		
	def getBackgroundFileList(self):
		if self.configReader.backgroundDirectoryPath != None:
			self.backgroundList = glob.glob(self.configReader.backgroundDirectoryPath+"/*."+self.configReader.imageFileExtension)
			self.backgroundList.sort() # or else their order is random...
		else:
			self.backgroundList = []
			
		self.totalNrOfBackgrounds = self.backgroundList.__len__()
		pass
	
	def getDarkfieldFileList(self):
		if self.configReader.darkfieldDirectoryPath != None:
			self.darkfieldList = glob.glob(self.configReader.darkfieldDirectoryPath+"/*."+self.configReader.imageFileExtension)
			self.darkfieldList.sort() # or else their order is random...
		else:
			self.darkfieldList = []
			
		self.totalNrOfDarkfields = self.darkfieldList.__len__()
		pass
	
	def printTrackingSetupInformation(self):
		print("")
		print("\tRunning Tracking with:")
		print("\tOpenCL Platform: "+self.clPlatformList[self.platformIndex].name)
		print("\tOpenCL Device: "+self.device.name)
		print("")
		print("\tDarkfield image directory:")
		if self.configReader.darkfieldDirectoryPath is not None:
			print("\t"+self.configReader.darkfieldDirectoryPath)
		else:
			print("\tNo directory provided.")
		print("")
		print("\tImage directory: ")
		print("\t"+self.configReader.imageDirectoryPath)
		print("")
		print("\tBackground image directory:")
		if self.configReader.backgroundDirectoryPath is not None:
			print("\t"+self.configReader.backgroundDirectoryPath)
		else:
			print("\tNo directory provided.")
		print("")
		print("\tSaving results to:")
		print("\t"+self.configReader.dataAnalysisDirectoryPath)
		print("")
	
	def saveTrackingResult(self):
		# for more info on usage see here: http://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html
		io.savemat(self.configReader.dataAnalysisDirectoryPath+'/contourCoordinatesX', mdict={'contourCoordinatesX': self.contourCoordinatesX},oned_as='row',do_compression=True)
		io.savemat(self.configReader.dataAnalysisDirectoryPath+'/contourCoordinatesY', mdict={'contourCoordinatesY': self.contourCoordinatesY},oned_as='row',do_compression=True)
		
		if self.configReader.nrOfFramesToSaveFitInclinesFor:
			io.savemat(self.configReader.dataAnalysisDirectoryPath+'/fitInclines', mdict={'fitInclines': self.fitInclines},oned_as='row',do_compression=True)

		if self.configReader.snrRoi is not None:
			io.savemat(self.configReader.dataAnalysisDirectoryPath+'/imageSnr', mdict={'imageSnr': self.imageSnr},oned_as='row',do_compression=True)
			io.savemat(self.configReader.dataAnalysisDirectoryPath+'/imageIntensity', mdict={'imageIntensity': self.imageIntensity},oned_as='row',do_compression=True)
		
		io.savemat(self.configReader.dataAnalysisDirectoryPath+'/contourNormalVectorsX', mdict={'contourNormalVectorsX': self.contourNormalVectorsX},oned_as='row',do_compression=True)
		io.savemat(self.configReader.dataAnalysisDirectoryPath+'/contourNormalVectorsY', mdict={'contourNormalVectorsY': self.contourNormalVectorsY},oned_as='row',do_compression=True)
		
		io.savemat(self.configReader.dataAnalysisDirectoryPath+'/contourCenterCoordinatesX', mdict={'contourCenterCoordinatesX': self.contourCenterCoordinatesX},oned_as='row',do_compression=True)
		io.savemat(self.configReader.dataAnalysisDirectoryPath+'/contourCenterCoordinatesY', mdict={'contourCenterCoordinatesY': self.contourCenterCoordinatesY},oned_as='row',do_compression=True)
		pass
		
	def writeContourToFinalArray(self,tracker):
		contourNr = tracker.getContourId()

		membraneCoordinatesX = tracker.getMembraneCoordinatesX()
		membraneCoordinatesY = tracker.getMembraneCoordinatesY()
		
		if self.nrOfFinishedImages < self.configReader.nrOfFramesToSaveFitInclinesFor:
			fitInclines = tracker.getFitInclines()
			self.fitInclines[:,contourNr] = fitInclines
		
		if self.configReader.snrRoi is not None:
			imageSnr = self.imagePreprocessor.getImageSnr()
			self.imageSnr[0,contourNr] = imageSnr
			imageIntensity = self.imagePreprocessor.getImageIntensity()
			self.imageIntensity[0,contourNr] = imageIntensity
			
		if np.any(np.isnan(membraneCoordinatesX)) or np.any(np.isnan(membraneCoordinatesY)):
			print("One or more contour coordinates are Nan. Saving results and aborting tracking.")
			self.saveTrackingResult()
			self.currentTime = time.time()
			runningTime = self.currentTime - self.startTime
			print("Total running time: "+str(datetime.timedelta(seconds=runningTime))+"h")
			
			ipdb.set_trace()
		
		self.contourCoordinatesX[:,contourNr] = membraneCoordinatesX
		self.contourCoordinatesY[:,contourNr] = membraneCoordinatesY
		
		contourNormalVectorsX = tracker.getMembraneNormalVectorsX()
		contourNormalVectorsY = tracker.getMembraneNormalVectorsY()
		self.contourNormalVectorsX[:,contourNr] = contourNormalVectorsX
		self.contourNormalVectorsY[:,contourNr] = contourNormalVectorsY
		
		contourCenter = tracker.getContourCenterCoordinates()

		self.contourCenterCoordinatesX[contourNr] = contourCenter['x'][0]
		self.contourCenterCoordinatesY[contourNr] = contourCenter['y'][0]
		pass
