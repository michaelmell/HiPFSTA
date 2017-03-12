import unittest
import os
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import matplotlib.pyplot as plt

class Test_testOpenClKernels(unittest.TestCase):
	def test_interpolatePolarCoordinatesLinear(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.setupClContext()
		self.loadClKernels()
		self.setupClQueue(self.ctx)
		path='C:/Private/PhD_Publications/Publication_of_Algorithm/Code/TrackingAlgorithm/TrackingAlgorithm/TestData/UnitTestData/interpolatePolarCoordinatesLinear/'
		self.nrOfLocalAngleSteps = 64
		self.detectionKernelStrideSize = 2048
		self.nrOfStrides = 1
		self.nrOfDetectionAngleSteps = np.float64(self.nrOfStrides*self.detectionKernelStrideSize)
		self.loadHostVariable('gradientGlobalSize',path)
		self.loadHostVariable('nrOfInterpolationPoints',path)
		self.loadHostVariable('nrOfDetectionAngleSteps',path)
		#self.loadHostVariable('nrOfAnglesToCompare',path)
		self.nrOfAnglesToCompare = np.int32(100)
		self.loadDeviceVariable('dev_membranePolarRadius',path)
		self.loadDeviceVariable('dev_membranePolarTheta',path)
		self.loadDeviceVariable('dev_radialVectors',path)
		self.loadDeviceVariable('dev_contourCenter',path)
		self.loadDeviceVariable('dev_membraneCoordinatesX',path)
		self.loadDeviceVariable('dev_membraneCoordinatesY',path)
		self.loadDeviceVariable('dev_interpolatedMembraneCoordinatesX',path)
		self.loadDeviceVariable('dev_interpolatedMembraneCoordinatesY',path)
		self.loadDeviceVariable('dev_membranePolarRadiusTMP',path)
		self.loadDeviceVariable('dev_membranePolarThetaTMP',path)
		self.loadDeviceVariable('dev_membranePolarRadiusInterpolation',path)
		self.loadDeviceVariable('dev_membranePolarThetaInterpolation',path)
		self.loadDeviceVariable('dev_membranePolarRadiusInterpolationTesting',path)
		self.loadDeviceVariable('dev_membranePolarThetaInterpolationTesting',path)
		self.loadDeviceVariable('dev_interpolationAngles',path)
		self.loadDeviceVariable("dev_dbgOut",path)
		self.loadDeviceVariable("dev_dbgOut2",path)
		self.setWorkGroupSizes()
		self.plotCurrentMembraneCoordinates()
		plt.show()

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

		self.plotCurrentInterpolatedMembraneCoordinates()
		plt.show()
		pass

	def loadHostVariable(self,variableName,path):
		host_tmp = np.load(path+'/'+variableName+'.npy')
		setattr(self,variableName,host_tmp)
		pass
	
	def loadDeviceVariable(self,variableName,path):
		host_tmp = np.load(path+'/'+variableName+'.npy')
		dev_tmp = cl_array.to_device(self.queue, host_tmp)
		setattr(self,variableName,dev_tmp)
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
		
	def loadClKernels(self):
		codePath = ".";
		modulePath = __file__
		codePath, filename = os.path.split(modulePath) # get location of path where our tracking code is located
		clCodeFile = codePath+"/"+"clKernelCode.cl"
		fObj = open(clCodeFile, 'r')
		self.kernelString = "".join(fObj.readlines())
		self.prg = cl.Program(self.ctx,self.kernelString).build()
		pass

	def setupClQueue(self,ctx):
		#~ self.ctx = cl.ctx([device])
		self.ctx = ctx
		self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
		self.mf = cl.mem_flags
	
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

	def plotCurrentMembraneCoordinates(self):
		host_membraneCoordinatesX = self.dev_membraneCoordinatesX.get()
		host_membraneCoordinatesY = self.dev_membraneCoordinatesY.get()
		plt.plot(host_membraneCoordinatesX,host_membraneCoordinatesY)

	def plotCurrentInterpolatedMembraneCoordinates(self):
		host_interpolatedMembraneCoordinatesX = self.dev_interpolatedMembraneCoordinatesX.get()
		host_interpolatedMembraneCoordinatesY = self.dev_interpolatedMembraneCoordinatesY.get()
		plt.plot(host_interpolatedMembraneCoordinatesX,host_interpolatedMembraneCoordinatesY)

if __name__ == '__main__':
	unittest.main()
