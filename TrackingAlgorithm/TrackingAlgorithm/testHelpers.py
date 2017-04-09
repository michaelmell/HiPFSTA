import unittest
import pyopencl as cl
import pyopencl.array as cl_array
from helpers import helpers
import numpy as np
from numpy import random as rnd

class Test_Helpers(unittest.TestCase):
	vectorLength = 10
	def test_ToDoubleVectorOnHost(self):
		singleVectorX = rnd.rand(self.vectorLength,1);
		singleVectorY = rnd.rand(self.vectorLength,1);
		doubleVector = helpers.ToDoubleVectorOnHost(singleVectorX,singleVectorY)
		self.assertTrue(np.all(doubleVector['x'] == singleVectorX[:,0]))
		self.assertTrue(np.all(doubleVector['y'] == singleVectorY[:,0]))

	def test_ToSingleVectorsOnHost(self):
		doubleVector = np.zeros(self.vectorLength, cl_array.vec.double2)
		xVals = rnd.rand(self.vectorLength,1);
		yVals = rnd.rand(self.vectorLength,1);
		doubleVector['x'] = xVals[:,0]
		doubleVector['y'] = yVals[:,0]
		xVector,yVector = helpers.ToSingleVectorsOnHost(doubleVector)
		self.assertTrue(np.all(xVector == xVals[:,0]))
		self.assertTrue(np.all(yVector == yVals[:,0]))

	def test_ToDoubleVectorOnDevice(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.setupClContext()
		self.setupClQueue(self.ctx)
		singleVectorX = rnd.rand(self.vectorLength,1);
		singleVectorY = rnd.rand(self.vectorLength,1);
		dev_singleVectorX = cl_array.to_device(self.queue,singleVectorX)
		dev_singleVectorY = cl_array.to_device(self.queue,singleVectorY)
		dev_doubleVector = helpers.ToDoubleVectorOnDevice(self.queue,dev_singleVectorX,dev_singleVectorY)
		doubleVector = dev_doubleVector.get()
		self.assertTrue(np.all(doubleVector['x'] == singleVectorX[:,0]))
		self.assertTrue(np.all(doubleVector['y'] == singleVectorY[:,0]))
		pass

	def test_ToSingleVectorsOnDevice(self):
		self.clPlatform = "intel"
		self.computeDeviceId = 0
		self.setupClContext()
		self.setupClQueue(self.ctx)

		doubleVector = np.zeros(self.vectorLength, cl_array.vec.double2)
		xVals = rnd.rand(self.vectorLength,1);
		yVals = rnd.rand(self.vectorLength,1);
		doubleVector['x'] = xVals[:,0]
		doubleVector['y'] = yVals[:,0]
		dev_doubleVector = cl_array.to_device(self.queue,doubleVector)

		dev_singleVectorX,dev_singleVectorY = helpers.ToSingleVectorsOnDevice(self.queue,dev_doubleVector)
		singleVectorX = dev_singleVectorX.get()
		singleVectorY = dev_singleVectorY.get()

		self.assertTrue(np.all(singleVectorX == xVals[:,0]))
		self.assertTrue(np.all(singleVectorY == yVals[:,0]))

		#singleVectorX = rnd.rand(self.vectorLength,1);
		#singleVectorY = rnd.rand(self.vectorLength,1);
		#dev_singleVectorX = cl_array.to_device(self.queue,singleVectorX)
		#dev_singleVectorY = cl_array.to_device(self.queue,singleVectorY)
		#dev_doubleVector = helpers.ToDoubleVectorOnDevice(self.queue,dev_singleVectorX,dev_singleVectorY)
		#doubleVector = dev_doubleVector.get()
		#self.assertTrue(np.all(doubleVector['x'] == singleVectorX[:,0]))
		#self.assertTrue(np.all(doubleVector['y'] == singleVectorY[:,0]))
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

	def setupClQueue(self,ctx):
		#~ self.ctx = cl.ctx([device])
		self.ctx = ctx
		self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
		self.mf = cl.mem_flags
	
if __name__ == '__main__':
	unittest.main()


