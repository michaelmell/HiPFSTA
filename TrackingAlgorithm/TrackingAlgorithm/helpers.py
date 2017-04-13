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
		return doubleVector

	def ToSingleVectorsOnHost(doubleVector):
		singleVectorX = np.copy(doubleVector['x'])
		singleVectorY = np.copy(doubleVector['y'])
		return singleVectorX, singleVectorY

	def ToDoubleVectorOnDevice(oclQueue,dev_singleVectorX,dev_singleVectorY):
		singleVectorX = dev_singleVectorX.get(oclQueue)
		singleVectorY = dev_singleVectorY.get(oclQueue)
		doubleVector = helpers.ToDoubleVectorOnHost(singleVectorX,singleVectorY)
		dev_doubleVector = cl_array.to_device(oclQueue,doubleVector)
		return dev_doubleVector

	def ToSingleVectorsOnDevice(oclQueue,dev_doubleVector):
		doubleVector = dev_doubleVector.get()
		singleVectorX,singleVectorY = helpers.ToSingleVectorsOnHost(doubleVector)
		dev_singleVectorX = cl_array.to_device(oclQueue,singleVectorX)
		dev_singleVectorY = cl_array.to_device(oclQueue,singleVectorY)
		return dev_singleVectorX,dev_singleVectorY
