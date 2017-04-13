import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import matplotlib.pyplot as plt

class helpers(object):
	"""description of class"""
	def __init__(self, **kwargs):
		return super().__init__(**kwargs)

	def ToDoubleVectorOnHost(singleVectorX,singleVectorY):
		vectorLength = singleVectorX.shape[0]
		doubleVector = np.zeros(vectorLength, cl_array.vec.double2)
		doubleVector['x'] = singleVectorX.copy()
		doubleVector['y'] = singleVectorY.copy()
		#differenceX = doubleVector['x']-singleVectorX
		#plt.plot(differenceX)
		#plt.show()
		return doubleVector

	def ToSingleVectorsOnHost(doubleVector):
		singleVectorX = np.copy(doubleVector['x'])
		singleVectorY = np.copy(doubleVector['y'])
		return singleVectorX, singleVectorY

	def ToDoubleVectorOnDevice(oclQueue,dev_singleVectorX,dev_singleVectorY):
		singleVectorX = dev_singleVectorX.get(oclQueue)
		#cl.enqueue_read_buffer(oclQueue, dev_singleVectorX.data, singleVectorX).wait()
		#barrierEvent = cl.enqueue_barrier(oclQueue)
		singleVectorY = dev_singleVectorY.get(oclQueue)
		#cl.enqueue_read_buffer(oclQueue, dev_singleVectorY.data, singleVectorY).wait()
		#barrierEvent = cl.enqueue_barrier(oclQueue)
		doubleVector = helpers.ToDoubleVectorOnHost(singleVectorX,singleVectorY)
		dev_doubleVector = cl_array.to_device(oclQueue,doubleVector)
		#barrierEvent = cl.enqueue_barrier(oclQueue)
		return dev_doubleVector

	def ToSingleVectorsOnDevice(oclQueue,dev_doubleVector):
		doubleVector = dev_doubleVector.get()
		singleVectorX,singleVectorY = helpers.ToSingleVectorsOnHost(doubleVector)
		dev_singleVectorX = cl_array.to_device(oclQueue,singleVectorX)
		dev_singleVectorY = cl_array.to_device(oclQueue,singleVectorY)
		return dev_singleVectorX,dev_singleVectorY
