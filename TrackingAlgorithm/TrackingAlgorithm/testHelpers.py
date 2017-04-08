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

if __name__ == '__main__':
	unittest.main()
