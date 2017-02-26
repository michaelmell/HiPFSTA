from contourTrackerClass import contourTracker
from sequentialContourTrackerClass import sequentialContourTracker

import ConfigParser
import json
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

class contourTrackerMain( object ):
	def __init__(self, configurationFile,runInteractive=True):
		self.runInteractive = runInteractive
		self.setInteractive(runInteractive)
		#~ self.configurationFile = configurationFile
		self.loadSettings(configurationFile)
		self.runConfigChecks()
		self.getImageFileList()
		self.getBackgroundFileList()
		self.setupClContext()
		self.setupManagementQueue()
		self.setupClVariables()
		self.printTrackingSetupInformation()
		self.setupTrackingQueues()
		
		self.contourCoordinatesX = np.empty((self.nrOfContourPoints,self.totalNrOfImages),dtype=np.float64)
		self.contourCoordinatesY = np.empty((self.nrOfContourPoints,self.totalNrOfImages),dtype=np.float64)
		
		self.contourNormalVectorsX = np.empty((self.nrOfContourPoints,self.totalNrOfImages),dtype=np.float64)
		self.contourNormalVectorsY = np.empty((self.nrOfContourPoints,self.totalNrOfImages),dtype=np.float64)
		
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

	def initializeTracking(self):
		if self.imageIndexToContinueFrom is 0:
			self.doInitialTracking()
		else:
			self.setupContinuationOfTracking()
		pass

	def setupClVariables(self):
		self.nrOfDetectionAngleSteps = self.detectionKernelStrideSize * self.nrOfStrides
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
		self.sequentialTracker.trackContour()
		
		if self.runInteractive:
			plt.imshow(self.sequentialTracker.host_Img)
			plt.plot(self.sequentialTracker.getMembraneCoordinatesX(),self.sequentialTracker.getMembraneCoordinatesY(),'k')
			plt.show()
		
		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneCoordinatesX.data,self.dev_mostRecentMembraneCoordinatesX.data).wait()
		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneCoordinatesY.data,self.dev_mostRecentMembraneCoordinatesY.data).wait()

		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneNormalVectorsX.data,self.dev_mostRecentMembraneNormalVectorsX.data).wait()
		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_membraneNormalVectorsY.data,self.dev_mostRecentMembraneNormalVectorsY.data).wait()

		cl.enqueue_copy_buffer(self.managementQueue,self.sequentialTracker.dev_contourCenter.data,self.dev_mostRecentContourCenter.data).wait()
		self.managementQueue.finish()
		
	def track(self):
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
		
		#~ iterationNr = 0
		#~ self.nrOfFinishedImages = 0
		while(self.nrOfFinishedImages<self.totalNrOfImages): # enter control-loop for checking an controlling the states of the tracking-queues
			for tracker in self.trackingQueues:
				#~ print "self.currentImageIndex: "+str(self.currentImageIndex)
				#~ print "self.mostRecentImageIndex: "+str(self.mostRecentImageIndex)
				#~ print "self.nrOfFinishedImages: "+str(self.nrOfFinishedImages)

				if tracker.iterationFinished:
					#~ if tracker.trackingFinished:
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
							#~ self.managementQueue.finish()
							self.mostRecentImageIndex = self.currentImageIndex
						
						frameId = tracker.getContourId()
						self.nrOfIterationsPerContour[frameId] = np.int32(tracker.getNrOfTrackingIterations())
						print "Nr of iterations: "+str(tracker.getNrOfTrackingIterations())
						self.executionTimePerContour[frameId] = np.float64(tracker.getExectionTime())
						print "Execution time: "+str(self.executionTimePerContour[frameId])+" sec"
						
						#~ if self.executionTimePerContour[frameId] > 2:
							#~ ipdb.set_trace()
							#~ plt.plot(tracker.getMembraneCoordinatesX(),tracker.getMembraneCoordinatesY())
							#~ plt.show()
						
						self.finishTime = time.time()
						runningTime = self.finishTime - self.startTime
						print "Total running time: "+str(datetime.timedelta(seconds=runningTime))+" h"
						print "\n"
						
						# do intermediate save points
						if self.mostRecentImageIndex % self.stepsBetweenSavingResults is 0:
							print "Saving intermediate results."
							print "\n"
							self.saveTrackingResult()
						
						# start tracking of new image
						if self.currentImageIndex < self.totalNrOfImages:
							tracker.resetNrOfTrackingIterations()
							tracker.startTimer()
							
							print "Tracking image: "+str(self.currentImageIndex+1)+" of "+str(self.totalNrOfImages) # 'self.currentImageIndex+1', because 'self.currentImageIndex' is zero-based index 
							print "Image File: "+os.path.basename(self.imageList[self.currentImageIndex]) # 'self.currentImageIndex+1', because 'self.currentImageIndex' is zero-based index 
							
							tracker.loadImage(self.imageList[self.currentImageIndex])
							tracker.setContourId(self.currentImageIndex)
							
							tracker.setStartingCoordinatesNew(self.dev_mostRecentMembraneCoordinatesX, \
															  self.dev_mostRecentMembraneCoordinatesY)
							tracker.setStartingMembraneNormals(self.dev_mostRecentMembraneNormalVectorsX, \
															   self.dev_mostRecentMembraneNormalVectorsY)
							
							#~ cl.enqueue_copy_buffer(self.managementQueue,self.dev_mostRecentContourCenter.data,tracker.dev_previousContourCenter.data).wait()
							
							tracker.trackContour()
							#~ tracker.queue.finish() # this is for debugging!! TODO: remove this
							self.currentImageIndex = self.currentImageIndex + 1
						
					else: # start new tracking iteration with the previous contour as starting position
						#~ print "Image File: "+self.imageList[self.currentImageIndex] # 'self.currentImageIndex+1', because 'self.currentImageIndex' is zero-based index 
						tracker.setStartingCoordinatesNew(tracker.dev_interpolatedMembraneCoordinatesX, \
														  tracker.dev_interpolatedMembraneCoordinatesY \
													     )
						#print "Nr of iterations: "+str(tracker.getNrOfTrackingIterations())
						
						#contourCenter = tracker.getContourCenterCoordinates() # TODO: remove this, once done DEBUGGING
						#print "centerX: "+str(contourCenter['x'][0])
						#print "centerY: "+str(contourCenter['y'][0])
						##~ xCenter = float(contourCenter['x'][0])
						##~ yCenter = float(contourCenter['y'][0])
						##~ ipdb.set_trace()
						##~ print "centerX: "+"{:3.7f}".format(xCenter)
						##~ print "centerY: "+"{:3.7f}".format(yCenter)
						
						tracker.trackContour()
						#~ tracker.queue.finish() # this is for debugging!! TODO: remove this
		
		# save fit results
		#~ ipdb.set_trace()
		print "Tracking finished. Saving results."
		self.saveTrackingResult()

		self.finishTime = time.time()
		runningTime = self.finishTime - self.startTime
		print "Total running time: "+str(datetime.timedelta(seconds=runningTime))+" h"
		
		pass
		
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
		
	def setupManagementQueue(self):
		self.managementQueue = cl.CommandQueue(self.ctx)
		self.mf = cl.mem_flags
		pass

	def setupTrackingQueues(self):
		#~ for index in xrange(nrOfTrackingQueues):
			#~ trackingQueues(index) = contourTracker( self.config )
		self.trackingQueues = [contourTracker(self.ctx, self.config) for count in xrange(self.nrOfTrackingQueues)]
		
		for tracker in self.trackingQueues:
			tracker.loadBackground(self.backgroundList) # load background images
		
		self.sequentialTracker = sequentialContourTracker(self.ctx, self.config)
		self.sequentialTracker.loadBackground(self.backgroundList)  # load background images
		
		#~ ipdb.set_trace()
		#~ trackingQueue[0].setupClQueue(self.ctx)
		pass

	def loadSettings(self,configurationFile):
		# check if configuration file exists
		#~ ipdb.set_trace()
		if os.path.isfile(configurationFile) is False:
			print "Error: Configuration file not found at: "+configurationFile
			sys.exit("0")
				
		self.config = ConfigParser.ConfigParser()
		self.config.read(configurationFile)
		
		#~ self.imagePath = json.loads(self.config.get("FileParameters","imagePath"))
		#~ self.backgroundDirectoryPath = json.loads(self.config.get("FileParameters","backgroundDirectoryPath"))
		backgroundDirectoryPath = self.config.get("FileParameters","backgroundDirectoryPath")
		if backgroundDirectoryPath == "" or backgroundDirectoryPath == "None":
			self.backgroundDirectoryPath = None
		else:
			self.backgroundDirectoryPath = json.loads(backgroundDirectoryPath)

		self.imageDirectoryPath = json.loads(self.config.get("FileParameters","imageDirectoryPath"))
		self.imageFileExtension = json.loads(self.config.get("FileParameters","imageFileExtension"))
		self.dataAnalysisDirectoryPath = json.loads(self.config.get("FileParameters","dataAnalysisDirectoryPath"))

		self.detectionKernelStrideSize = np.int32(json.loads(self.config.get("TrackingParameters","detectionKernelStrideSize")))
		self.nrOfStrides = np.int32(json.loads(self.config.get("TrackingParameters","nrOfStrides")))
		self.nrOfContourPoints = self.detectionKernelStrideSize*self.nrOfStrides
		
		self.clPlatform = json.loads(self.config.get("OpenClParameters","clPlatform"))
		self.computeDeviceId = json.loads(self.config.get("OpenClParameters","computeDeviceId"))
		self.nrOfTrackingQueues = json.loads(self.config.get("OpenClParameters","nrOfTrackingQueues"))
		
		self.detectionKernelStrideSize = np.int32(json.loads(self.config.get("TrackingParameters","detectionKernelStrideSize")))
		self.nrOfStrides = np.int32(json.loads(self.config.get("TrackingParameters","nrOfStrides")))
		
		self.imageIndexToContinueFrom = json.loads(self.config.get("TrackingParameters","imageIndexToContinueFrom"))
		
		self.stepsBetweenSavingResults = json.loads(self.config.get("TrackingParameters","stepsBetweenSavingResults"))
		
		#~ dict = self.config.items("SectionOne")
		
		#~ ipdb.set_trace()

		#~ for item in dict:
			#~ print "self.configuration."+item[0]+"="+item[1]
			#~ eval("self.configuration."+item[0]+"="+item[1])
		#~ ipdb.set_trace()
		
		pass

	def getImageFileList(self):
		#~ self.imageList = glob.glob(self.imagePath+"/*.tif")
		self.imageList = glob.glob(self.imageDirectoryPath+"/*."+self.imageFileExtension)
		self.imageList.sort() # or else their order is random...
		self.totalNrOfImages = self.imageList.__len__()
		#~ ipdb.set_trace()
		pass
		
	def getBackgroundFileList(self):
		if self.backgroundDirectoryPath != None:
		#~ self.imageList = glob.glob(self.imagePath+"/*.tif")
			self.backgroundList = glob.glob(self.backgroundDirectoryPath+"/*."+self.imageFileExtension)
			self.backgroundList.sort() # or else their order is random...
		else:
			self.backgroundList = []
			
		self.totalNrOfBackgrounds = self.backgroundList.__len__()
		#~ ipdb.set_trace()
		pass
	
	def runConfigChecks(self):
		if self.backgroundDirectoryPath is not None:
			if not os.path.isdir(self.backgroundDirectoryPath):
				print ""
				print "\tERROR: Directory at 'backgroundDirectoryPath' does not exist."
				exit()
		if not os.path.isdir(self.imageDirectoryPath):
			print ""
			print "\tERROR: Directory at 'imageDirectoryPath' does not exist."
			exit()
		if not os.path.isdir(self.dataAnalysisDirectoryPath):
			if self.runInteractive: # if we are not running interactive, we know what we're doing (e.g. overwriting by default)
				print ""
				print "\tWARNING: Directory at 'dataAnalysisDirectoryPath' does not exist. Create it?"
				print ""
				answer = raw_input("\tContinue? (y: yes, n: no) ")
				if answer.lower().startswith("y"):
					os.makedirs(self.dataAnalysisDirectoryPath)
				else:
					exit()
			else:
				print ""
				print "\tWARNING: Directory at 'dataAnalysisDirectoryPath' did not exist. Created it."
				print ""
				os.makedirs(self.dataAnalysisDirectoryPath)

		if os.listdir(self.dataAnalysisDirectoryPath) != [] and self.imageIndexToContinueFrom == 0:
			if self.runInteractive: # if we are not running interactive, we know what we're doing (e.g. overwriting by default)
				print ""
				print "\tWARNING: Directory at 'dataAnalysisDirectoryPath' is not empty and 'imageIndexToContinueFrom' is 0. Continuing may result in data-loss."
				print ""
				print "\t'dataAnalysisDirectoryPath':"
				print "\t"+self.dataAnalysisDirectoryPath
				print ""
				answer = raw_input("\tContinue? (y: yes, n: no) ")
				if answer.lower().startswith("n"):
					exit()
	
	def printTrackingSetupInformation(self):
		print ""
		print "\tRunning Tracking with:"
		print "\tOpenCL Platform: "+self.clPlatformList[self.platformIndex].name
		print "\tOpenCL Device: "+self.device.name
		print ""
		print "\tImage directory: "
		print "\t"+self.imageDirectoryPath
		print ""
		if self.backgroundDirectoryPath is not None:
			print "\tBackground image directory:"
			print "\t"+self.backgroundDirectoryPath
		else:
			print "\tBackground image directory:"
			print "\tNo directory provided."
		print ""
		print "\tSaving results to:"
		print "\t"+self.dataAnalysisDirectoryPath
		print ""
	
	def saveTrackingResult(self):
		#~ ipdb.set_trace()
		#~ self.dataAnalysisDirectoryPath
		#~ io.savemat(self.dataAnalysisDirectoryPath+'/contourCoordinatesX', mdict={'contourCoordinatesX': self.contourCoordinatesX})
		#~ io.savemat(self.dataAnalysisDirectoryPath+'/contourCoordinatesY', mdict={'contourCoordinatesY': self.contourCoordinatesY})
		# for more info on usage see here: http://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html
		io.savemat(self.dataAnalysisDirectoryPath+'/contourCoordinatesX', mdict={'contourCoordinatesX': self.contourCoordinatesX},oned_as='row',do_compression=True)
		io.savemat(self.dataAnalysisDirectoryPath+'/contourCoordinatesY', mdict={'contourCoordinatesY': self.contourCoordinatesY},oned_as='row',do_compression=True)
		
		io.savemat(self.dataAnalysisDirectoryPath+'/contourNormalVectorsX', mdict={'contourNormalVectorsX': self.contourNormalVectorsX},oned_as='row',do_compression=True)
		io.savemat(self.dataAnalysisDirectoryPath+'/contourNormalVectorsY', mdict={'contourNormalVectorsY': self.contourNormalVectorsY},oned_as='row',do_compression=True)
		
		io.savemat(self.dataAnalysisDirectoryPath+'/contourCenterCoordinatesX', mdict={'contourCenterCoordinatesX': self.contourCenterCoordinatesX},oned_as='row',do_compression=True)
		io.savemat(self.dataAnalysisDirectoryPath+'/contourCenterCoordinatesY', mdict={'contourCenterCoordinatesY': self.contourCenterCoordinatesY},oned_as='row',do_compression=True)
		
		# DEBUGGING; TODO: REMOVE THIS LATER...
		#~ ipdb.set_trace()
		#~ tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/contourCoordinatesX_REF.mat')
		#~ contourCoordinatesX_REF = tmp['contourCoordinatesX']
		#~ tmp=io.loadmat(self.dataAnalysisDirectoryPath+'/contourCoordinatesY_REF.mat')
		#~ contourCoordinatesY_REF = tmp['contourCoordinatesY']
		
		#~ if np.all(contourCoordinatesX_REF==self.contourCoordinatesX) and np.all(contourCoordinatesY_REF==self.contourCoordinatesY):
		#~ if np.allclose(contourCoordinatesX_REF,self.contourCoordinatesX) and np.allclose(contourCoordinatesY_REF,self.contourCoordinatesY):
			#~ print "Coordinate test PASSED."
		#~ else:
			#~ print "Coordinate test FAILED."
			
		pass
		
	def writeContourToFinalArray(self,tracker):
		contourNr = tracker.getContourId()
		#~ ipdb.set_trace()
		#~ self.contourCoordinatesX[:,contourNr] = tracker.getMembraneCoordinatesX()
		#~ self.contourCoordinatesY[:,contourNr] = tracker.getMembraneCoordinatesY()
		membraneCoordinatesX = tracker.getMembraneCoordinatesX()
		membraneCoordinatesY = tracker.getMembraneCoordinatesY()
		
		if np.any(np.isnan(membraneCoordinatesX)) or np.any(np.isnan(membraneCoordinatesY)):
			print "One or more contour coordinates are Nan. Saving results and aborting tracking."
			self.saveTrackingResult()
			self.finishTime = time.time()
			runningTime = self.finishTime - self.startTime
			print "Total running time: "+str(datetime.timedelta(seconds=runningTime))+"h"
			
			ipdb.set_trace()
		
		self.contourCoordinatesX[:,contourNr] = membraneCoordinatesX
		self.contourCoordinatesY[:,contourNr] = membraneCoordinatesY
		
		contourNormalVectorsX = tracker.getMembraneNormalVectorsX()
		contourNormalVectorsY = tracker.getMembraneNormalVectorsY()
		self.contourNormalVectorsX[:,contourNr] = contourNormalVectorsX
		self.contourNormalVectorsY[:,contourNr] = contourNormalVectorsY
		
		contourCenter = tracker.getContourCenterCoordinates()
		#~ ipdb.set_trace()
		self.contourCenterCoordinatesX[contourNr] = contourCenter['x'][0]
		self.contourCenterCoordinatesY[contourNr] = contourCenter['y'][0]
				
		
		#~ if np.any(np.isnan(self.contourCoordinatesX)) or np.any(np.isnan(self.contourCoordinatesY)):
			#~ ipdb.set_trace()
		
		#~ if contourNr == 1:
		#~ if contourNr > self.contourCoordinatesX.shape[1]-10:
			#~ ipdb.set_trace()
			#~ 
			#~ import matplotlib.pyplot as plt
			#~ 
			#~ plt.plot(self.contourCoordinatesX[:,0:2],self.contourCoordinatesY[:,0:2])
			#~ plt.show()
			
			#~ ax = plt.gca()
			#~ nrOfContourPoints = self.contourCoordinatesX.shape[0]
			#~ contourCentersX = np.tile(self.contourCenterCoordinatesX,[nrOfContourPoints,1])
			#~ contourCentersY = np.tile(self.contourCenterCoordinatesY,[nrOfContourPoints,1])
			#~ plt.plot(self.contourCoordinatesX[:,0:20]-contourCentersX[:,0:20],self.contourCoordinatesY[:,0:20]-contourCentersY[:,0:20])
			#~ plt.show()
			
			#~ ax = plt.gca()
			#~ selectedContourNr = 6
			#~ nrOfContourPoints = self.contourCoordinatesX.shape[0]
			#~ contourCentersX = np.tile(self.contourCenterCoordinatesX,[nrOfContourPoints,1])
			#~ contourCentersY = np.tile(self.contourCenterCoordinatesY,[nrOfContourPoints,1])
			#~ plt.plot(self.contourCoordinatesX[:,selectedContourNr]-contourCentersX[:,selectedContourNr],self.contourCoordinatesY[:,selectedContourNr]-contourCentersY[:,selectedContourNr])
			#~ plt.show()

		pass
