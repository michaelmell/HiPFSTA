import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

class helpers(object):
	"""description of class"""
	def __init__(self, **kwargs):
		return super().__init__(**kwargs)

	def ToDoubleVectorOnHost(singleVectorX,singleVectorY):
		vectorLength = singleVectorX.shape[0]
		doubleVector = np.zeros(vectorLength, cl_array.vec.double2)
		doubleVector['x'] = singleVectorX[:,0]
		doubleVector['y'] = singleVectorY[:,0]
		return doubleVector

	def ToSingleVectorsOnHost(doubleVector):
		singleVectorX = doubleVector['x']
		singleVectorY = doubleVector['y']
		return singleVectorX, singleVectorY
