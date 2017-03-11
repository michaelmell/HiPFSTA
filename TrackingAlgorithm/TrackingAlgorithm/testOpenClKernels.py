import unittest

class Test_testOpenClKernels(unittest.TestCase):
	def test_interpolatePolarCoordinatesLinear(self):
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
		pass

	def loadClKernels(self):
		modulePath = __file__
		codePath, filename = os.path.split(modulePath) # get location of path where our tracking code is located
		clCodeFile = codePath+"/"+"clKernelCode.cl"
		fObj = open(clCodeFile, 'r')
		self.kernelString = "".join(fObj.readlines())
		self.prg = cl.Program(self.ctx,self.kernelString).build()
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

if __name__ == '__main__':
	unittest.main()
