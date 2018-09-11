#define lineIntensities_LENGTH ${lineIntensities_LENGTH}

#define MIN_MAX_INTENSITY_SEARCH 1
#define MAX_INCLINE_SEARCH 2

#define LINEAR_FIT_SEARCH_METHOD ${linear_fit_search_method}


#if LINEAR_FIT_SEARCH_METHOD == MIN_MAX_INTENSITY_SEARCH
	#define LINEAR_FIT_SEARCH_METHOD_CALL() determineFitUsingMinMaxIntensitySearch(lineIntensities,lineIntensities_LENGTH,linFitParameter,linFitSearchRangeXvalues)
#elif LINEAR_FIT_SEARCH_METHOD == MAX_INCLINE_SEARCH
	#define LINEAR_FIT_SEARCH_METHOD_CALL() determineFitUsingInclineSearch(lineIntensities,lineIntensities_LENGTH,linFitParameter,linFitSearchRangeXvalues,inclineRefinementRange)
#endif


#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif
//~ #pragma OPENCL EXTENSION cl_amd_printf : enable

void linearFit(__constant double x[], double y[], int gradientCenterIndex, int linFitParameter, double *a, double *b, double *siga, double *sigb, double *chi2)
{
	int i;
	double t,sxoss,sx=0.0,sy=0.0,st2=0.0,ss,sigdat;
	*b=0.0;

	int ndata = 2*linFitParameter;
	int startInd = gradientCenterIndex-linFitParameter;
	int endInd = gradientCenterIndex+linFitParameter;
	
	for (i=startInd;i<endInd;i++){ // ...or without weights.
		sx += x[i];
		sy += y[i];
	}
	ss=ndata;
	sxoss=sx/ss;
	
	for (i=startInd;i<endInd;i++){
		t=x[i]-sxoss;
		st2 += t*t;
		*b += t*y[i];
	}
	
	*b /= st2; // Solve for a, b, σa,and σb.
	*a=(sy-sx*(*b))/ss;
	*siga=sqrt((1.0+sx*sx/(ss*st2))/ss);
	*sigb=sqrt(1.0/st2);
	
	*chi2=0.0; // Calculate χ2.
	
	for (i=startInd;i<endInd;i++){
		*chi2 += sqrt(y[i]-(*a)-(*b)*x[i]);
	}

	sigdat=sqrt((*chi2)/(ndata-2)); // For unweighted data evaluate typical sig using chi2,and adjust the standard deviations.
	*siga *= sigdat;
	*sigb *= sigdat;
}

/* This modulo function returns the modulo value index%arraySize, where index is always positive, with value 
 * 0<=index<arraySize-1. This allows use to easily perform circular indexing of our arrays.
 * For further details on this solution see here, where I found it:
 * https://codereview.stackexchange.com/questions/57923/index-into-array-as-if-it-is-circular
 * A similar solution can be found here:
 * http://stackoverflow.com/questions/4868049/how-to-efficiently-wrap-the-index-of-a-fixed-size-circular-buffer
 */
int WrapIndex(int index, int arraySize)
{
  return (index % arraySize + arraySize) % arraySize;
}

__kernel void calculateMembraneNormalVectors(__global double2* membraneCoordinates,
											 __global double2* membraneNormalVectors
											)
	{
		const int arraySize = get_global_size(0);
		const int currentInd = get_global_id(0);
		const int prevNeighborInd = WrapIndex(currentInd-1,arraySize);
		const int nextNeighborInd = WrapIndex(currentInd+1,arraySize);
		
		membraneNormalVectors[currentInd].y = -(  (membraneCoordinates[currentInd].x - membraneCoordinates[prevNeighborInd].x)
												+ (membraneCoordinates[nextNeighborInd].x - membraneCoordinates[currentInd].x) )/2;
		membraneNormalVectors[currentInd].x =  (  (membraneCoordinates[currentInd].y - membraneCoordinates[prevNeighborInd].y)
												+ (membraneCoordinates[nextNeighborInd].y - membraneCoordinates[currentInd].y) )/2;
		
		membraneNormalVectors[currentInd] = normalize(membraneNormalVectors[currentInd]);
	}

__kernel void calculateDs(
						 __global double2* membraneCoordinates,
						 __global double* ds
						 )
{
	const int arraySize = get_global_size(0);
	const int currentInd = get_global_id(0);
	const int prevNeighborInd = WrapIndex(currentInd-1,arraySize);
	
	ds[currentInd] = length(membraneCoordinates[currentInd]-membraneCoordinates[prevNeighborInd]);
}

__kernel void calculateSumDs(__global double* ds,
							 __global double* sumds
							 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	if(xInd>=1){
		sumds[xInd] = ds[xInd] + ds[xInd-1];
	}
	else if(xInd==0){
		sumds[xInd] = ds[xInd] + ds[xSize-1];
	}
}

__kernel void calculateContourCenter(__global double2* membraneCoordinates,
									 __global double* ds,
									 __global double* sumds,
									 __global double2* contourCenter,
									 const int nrOfContourPoints
									 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	__private double circumference=0.0;
	
	__private double tmp1=0.0, tmp2=0.0;
	if(xInd==0){
		for(int index=0;index<nrOfContourPoints;index++){
			circumference = circumference + ds[index];
		}
		
		for(int index=0;index<nrOfContourPoints;index++){
			tmp1 = tmp1 + membraneCoordinates[index].x * sumds[index];
			tmp2 = tmp2 + membraneCoordinates[index].y * sumds[index];
		}
		contourCenter[0].x = (1/(2*circumference)) * tmp1;
		contourCenter[0].y = (1/(2*circumference)) * tmp2;
	}
}

__kernel void cart2pol(__global double2* membraneCoordinates,
					   __global double2* membranePolarCoordinates,
					   __global double2* contourCenter
					  )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	double2 contourCenterLoc;
	contourCenterLoc = contourCenter[0];
	
	membranePolarCoordinates[xInd].x =  atan2( membraneCoordinates[xInd].y - contourCenterLoc.y,
												membraneCoordinates[xInd].x - contourCenterLoc.x);
	membranePolarCoordinates[xInd].y =  sqrt( pow((membraneCoordinates[xInd].y - contourCenterLoc.y),2)
											 + pow((membraneCoordinates[xInd].x - contourCenterLoc.x),2) 
											 );
}

__kernel void pol2cart(__global double2* membraneCoordinates,
					   __global double2* membranePolarCoordinates,
					   __global double2* contourCenter
					  )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	double2 contourCenterLoc;
	contourCenterLoc = contourCenter[0];
	
	membraneCoordinates[xInd].x = contourCenterLoc.x + membranePolarCoordinates[xInd].y * cos(membranePolarCoordinates[xInd].x);
	membraneCoordinates[xInd].y = contourCenterLoc.y + membranePolarCoordinates[xInd].y * sin(membranePolarCoordinates[xInd].x);
}

__kernel void emptyKernel(__global double* membraneCoordinatesX, __global double* membraneCoordinatesY)
{
	const int xInd = get_global_id(1);
	const int yInd = get_global_id(0);
	const int xSize = get_global_size(1);
	const int ySize = get_global_size(0);

	const int xIndLoc = get_local_id(1);
	const int yIndLoc = get_local_id(0);
	
	membraneCoordinatesX[xInd] = membraneCoordinatesX[xInd];
	membraneCoordinatesY[xInd] = membraneCoordinatesY[xInd];
}

/* Sets self.dev_iterationFinished to true, when called */
__kernel void setIterationFinished(__global int* iterationFinished)
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	iterationFinished[0] = 1;
}

/* 
 * This function will set trackingFinished to 0 (FALSE),
 * if the euclidian distance between any coordinate of the
 * interpolated contours is >coordinateTolerance
 */
__kernel void checkIfTrackingFinished(
									__global double2* interpolatedMembraneCoordinates,
									__global double2* previousInterpolatedMembraneCoordinates,
									__global int* trackingFinished,
									const double coordinateTolerance
									)
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	__private double distance =  length(interpolatedMembraneCoordinates[xInd] - previousInterpolatedMembraneCoordinates[xInd]);
	
	if(distance>coordinateTolerance)
	{
		trackingFinished[0] = 0;
	}
}

/**********************************************************************
 * This function sorts the coordinates according the angle corresponding 
 * to each entry of the other arrays ()
 * It uses the bubble sort algorithm described on Wikipedia
 * (http://en.wikipedia.org/wiki/Bubble_sort):
 * 
 * 	procedure bubbleSort( A : list of sortable items )
 *		n = length(A)
 *		repeat
 *		   newn = 0
 *		   for i = 1 to n-1 inclusive do
 *			  if A[i-1] > A[i] then
 *				 swap(A[i-1], A[i])
 *				 newn = i
 *			  end if
 *		   end for
 *		   n = newn
 *		until n = 0
 *	end procedure
 * 
 * ********************************************************************/
__kernel void sortCoordinates(__global double2* membranePolarCoordinates,
							  __global double2* membraneCoordinates,
							  __global double2* membraneNormalVectors,
							  const int nrOfContourPoints
							  )
{
	const int xInd = get_global_id(1);
	const int yInd = get_global_id(0);
	const int xSize = get_global_size(1);
	const int ySize = get_global_size(0);

	if(xInd==0 && yInd==0)
	{
		int n = nrOfContourPoints;
		while(n!=0)
		{
		   int newn = 0;
		   for(int i=1;i<=n-1;i++)
		   {
				if(membranePolarCoordinates[i-1].x>membranePolarCoordinates[i].x)
				{
					__private double2 tmp;
					
					tmp = membranePolarCoordinates[i-1];
					membranePolarCoordinates[i-1] = membranePolarCoordinates[i];
					membranePolarCoordinates[i] = tmp;
					
					tmp = membraneCoordinates[i-1];
					membraneCoordinates[i-1] = membraneCoordinates[i];
					membraneCoordinates[i] = tmp;

					tmp = membraneNormalVectors[i-1];
					membraneNormalVectors[i-1] = membraneNormalVectors[i];
					membraneNormalVectors[i] = tmp;
					newn = i;
				}
		   }
		   n = newn;
	   }
	}
}

double2 calculateLocalMembranePositions(double fitIncline, double fitIntercept, double meanIntensity, double2 basePoint, double2 unitVector)
{
	__private double relativeMembranePositionLocalCoordSys;

	if(fitIncline != 0)
	{
		relativeMembranePositionLocalCoordSys = (meanIntensity-fitIntercept)/fitIncline;
	}
	else
	{
		relativeMembranePositionLocalCoordSys = 0;
	}

	return basePoint + unitVector * relativeMembranePositionLocalCoordSys;
	
}

double getImageIntensitiesAtCoordinate(image2d_t Img, sampler_t sampler, double2 Coords)
{
	__private double2 NormCoords;
	__private const int2 dims = get_image_dim(Img);

	NormCoords = Coords/convert_double2(dims);
		
	float2 fNormCoords;
	fNormCoords = convert_float2(NormCoords);
	
	return read_imagef(Img, sampler, fNormCoords).x;
}

typedef struct linearFitResultStruct
{
	double fitIntercept;
	double fitIncline;
} linearFitResult;

struct linearFitResultStruct determineFitUsingMinMaxIntensitySearch(double lineIntensities[], const int lineIntensitiesLength, const int linFitParameter, __constant double linFitSearchRangeXvalues[])
{
	__private int maxIndex;
	__private int minIndex;
	__private double minValue = 32000; // initialize value with first value of array
	__private double maxValue = 0; // initialize value with first value of array
	
	for(int index=0;index<lineIntensitiesLength;index++)
	{
		maxIndex = select(maxIndex,index,(maxValue < lineIntensities[index]));
		maxValue = select(maxValue,lineIntensities[index],(long)(maxValue < lineIntensities[index]));

		minIndex = select(minIndex,index,(minValue > lineIntensities[index]));
		minValue = select(minValue,lineIntensities[index],(long)(minValue > lineIntensities[index]));
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	__private int gradientCenterIndex;
	__private double gradientCenterValue = minValue+(maxValue-minValue)/2.0;
	__private double minValue2 = 20000;
	__private double refValue;
	
	__private double a=0.0, b=0.0, siga=0.0, sigb=0.0, chi2=0.0;
		
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	__private struct linearFitResultStruct linearFitResult;
	
	if(minIndex<maxIndex)
	{
		for(int index=minIndex;index<maxIndex;index++)
		{
			refValue = fabs(lineIntensities[index]-gradientCenterValue);
			gradientCenterIndex = select(gradientCenterIndex,index,(minValue2 > refValue));
			minValue2 = select(minValue2,refValue,(long)(minValue2 > refValue));
		}
		// "r = a > b ? a : b" corresponds to: "r = select(b, a, a > b)", corresponds to "if(a>b){r=a}; else{r=b}"
		// reference: http://stackoverflow.com/questions/7635706/opencl-built-in-function-select
		
		linearFit(linFitSearchRangeXvalues, lineIntensities, gradientCenterIndex, linFitParameter, &a, &b, &siga, &sigb, &chi2);
		linearFitResult.fitIntercept = a;
		linearFitResult.fitIncline = b;
	}
	else
	{
		linearFitResult.fitIntercept = 0; // so that later they are not counted in the weighted sum (see below...)
		linearFitResult.fitIncline = 0;
	}

	return linearFitResult;
}

struct linearFitResultStruct determineFitUsingInclineSearch(double lineIntensities[], const int lineIntensitiesLength, const int linFitParameter, __constant double linFitSearchRangeXvalues[], const int inclineRefinementRange)
{
	__private double a=0.0, b=0.0, siga=0.0, sigb=0.0, chi2=0.0;
	__private double aTmp=0.0, bTmp=0.0, bTmp2;
	
	int intervalCenterIndex = lineIntensitiesLength/2;
	int lowerLimit = intervalCenterIndex - inclineRefinementRange;
	int upperLimit = intervalCenterIndex + inclineRefinementRange;
	
	for(int index=lowerLimit; index<upperLimit; index++)
	{
		linearFit(linFitSearchRangeXvalues, lineIntensities, index, linFitParameter, &a, &b, &siga, &sigb, &chi2);
		bTmp2 = bTmp;
		bTmp = select(bTmp,b,(long)(fabs(b) > bTmp2));
		aTmp = select(aTmp,a,(long)(fabs(b) > bTmp2));
	}

	__private struct linearFitResultStruct linearFitResult;
	linearFitResult.fitIntercept = aTmp;
	linearFitResult.fitIncline = bTmp;
	return linearFitResult;
}

//~ #pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void findMembranePosition(sampler_t sampler, 
								   __read_only image2d_t Img,
								   const int imgSizeX,
								   const int imgSizeY,
								   __constant double* localRotationMatrices, // this should be local or constant
								   __constant double* linFitSearchRangeXvalues, // this should be local or constant
								   const int linFitParameter,
								   __local double* fitIntercept,
								   __local double* fitIncline,
								   __local double2* rotatedUnitVector2,
								   const int meanParameter,
								   __constant double* meanRangeXvalues, // this should be local or constant
								   const double meanRangePositionOffset,
								   __local double2* localMembranePositions,
								   __global double2* membraneCoordinates,
								   __global double2* membraneNormalVectors,
								   __global double* fitInclines,
								   const int coordinateStartingIndex,
								   const double inclineTolerance,
								   const int inclineRefinementRange)
{
	const int xInd = get_global_id(1);
	const int yInd = get_global_id(0);
	const int xSize = get_global_size(1);
	const int ySize = get_global_size(0);
	
	const int xGroupId = get_group_id(1);
	const int yGroupId = get_group_id(0);
	
	const int xIndLoc = get_local_id(1);
	const int yIndLoc = get_local_id(0);
	const int xSizeLoc = get_local_size(1);
	const int ySizeLoc = get_local_size(0);
	
	const int coordinateIndex = coordinateStartingIndex + yGroupId*ySizeLoc + yIndLoc;
	
	__private double lineIntensities[lineIntensities_LENGTH];
	
	__private double2 membraneNormalVector = membraneNormalVectors[coordinateIndex];
	
	// matrix multiplication with linear array of sequential 2x2 rotation matrices
	rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].x = localRotationMatrices[4*xIndLoc+0] * membraneNormalVector.x
												   + localRotationMatrices[4*xIndLoc+1] * membraneNormalVector.y;
	rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].y = localRotationMatrices[4*xIndLoc+2] * membraneNormalVector.x
												   + localRotationMatrices[4*xIndLoc+3] * membraneNormalVector.y;

	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	__private double2 Coords;
	
	__private double2 basePoint = membraneCoordinates[coordinateIndex];
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int index=0;index<lineIntensities_LENGTH;index++)
	{
		Coords = basePoint + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc] * linFitSearchRangeXvalues[index];
		lineIntensities[index] = getImageIntensitiesAtCoordinate(Img, sampler, Coords);
	}
		
	__private struct linearFitResultStruct fitResult = LINEAR_FIT_SEARCH_METHOD_CALL();
	fitIntercept[xIndLoc+yIndLoc*xSizeLoc] = fitResult.fitIntercept;
	fitIncline[xIndLoc+yIndLoc*xSizeLoc] = fitResult.fitIncline;
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	__private double meanIntensity = 0.0;
	for(int index=0;index<meanParameter;index++)
	{
		Coords = basePoint + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc] * ( meanRangeXvalues[index] + meanRangePositionOffset );
		meanIntensity += getImageIntensitiesAtCoordinate(Img, sampler, Coords);
	}
	meanIntensity = meanIntensity/convert_float(meanParameter);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	localMembranePositions[xIndLoc+yIndLoc*xSizeLoc] = calculateLocalMembranePositions(fitIncline[xIndLoc+yIndLoc*xSizeLoc],
																					   fitIntercept[xIndLoc+yIndLoc*xSizeLoc],
																					   meanIntensity,
																					   basePoint,
																					   rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc]);
	
	write_mem_fence(CLK_LOCAL_MEM_FENCE);

	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	/* *****************************************************************
	 * Find largest inclination value in workgroup and save to maxFitIncline
	 * ****************************************************************/
	__local double maxFitIncline;

	if(xIndLoc==0 & yIndLoc==0)
	{
		maxFitIncline = 0.0;
		for(int index=0;index<xSizeLoc*ySizeLoc;index++)
		{
			if(fabs(fitIncline[index])>fabs(maxFitIncline))
			{
				maxFitIncline = fitIncline[index];
			}
		}
	}

	write_mem_fence(CLK_LOCAL_MEM_FENCE);

	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	
	__private double2 CoordTmp = {0.0f,0.0f};
	__private double inclineSum = 0.0;
	__private double2 membraneNormalTmp = {0.0f,0.0f};
	
	if(xIndLoc==0)
	{
			for(int index=0;index<xSize;index++)
			{
				if(fabs(fitIncline[index+yIndLoc*xSizeLoc])>inclineTolerance*fabs(maxFitIncline))
				{
					CoordTmp += fabs(fitIncline[index+yIndLoc*xSizeLoc]) * localMembranePositions[index+yIndLoc*xSizeLoc];
					
					membraneNormalTmp += fabs(fitIncline[index+yIndLoc*xSizeLoc]) * rotatedUnitVector2[index+yIndLoc*xSizeLoc];
					
					inclineSum += fabs(fitIncline[index+yIndLoc*xSizeLoc]);
				}
			}
			
			membraneCoordinates[coordinateIndex] = CoordTmp/inclineSum;
			fitInclines[coordinateIndex] = maxFitIncline;
			
			membraneNormalTmp = membraneNormalTmp/inclineSum;
			__private double membraneNormalNorm = length(membraneNormalTmp);
			membraneNormalVectors[coordinateIndex] = membraneNormalTmp/membraneNormalNorm;
	}
}

__kernel void filterNanValues(__global double2* membraneCoordinates,
							  __global double2* membraneNormalVectors,
							  __local int* closestLowerNoneNanIndexLoc,
							  __local int* closestUpperNoneNanIndexLoc
							 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	__private bool NanValueLeft;

	closestLowerNoneNanIndexLoc[xInd] = xInd;
	closestUpperNoneNanIndexLoc[xInd] = xInd;

	__private int distToLowerIndex = 0;
	__private int distToUpperIndex = 0;
	do
	{
		NanValueLeft = false;
		if(isnan(membraneCoordinates[closestLowerNoneNanIndexLoc[xInd]].x)) // TODO: Why do we not check the y-coordinate for NaN values?! - Michael 2017-04-15
		{
			closestLowerNoneNanIndexLoc[xInd] -= 1;
			distToLowerIndex++;
			NanValueLeft = true;
		}

		if(isnan(membraneCoordinates[closestUpperNoneNanIndexLoc[xInd]].x)) // TODO: Why do we not check the y-coordinate for NaN values?! - Michael 2017-04-15
		{
			closestUpperNoneNanIndexLoc[xInd] += 1;
			distToUpperIndex++;
			NanValueLeft = true;
		}
		if(closestLowerNoneNanIndexLoc[xInd]<0) // avoid that we round out array bounds by using periodic boundaries
		{
			closestLowerNoneNanIndexLoc[xInd] = closestLowerNoneNanIndexLoc[xInd]+xSize;
		}
		if(closestUpperNoneNanIndexLoc[xInd]>xSize-1) // avoid that we round out array bounds by using periodic boundaries
		{
			closestUpperNoneNanIndexLoc[xInd] = closestUpperNoneNanIndexLoc[xInd]-xSize;
		}
	}
	while(NanValueLeft);

	/* *****************************************************************
	 * interpolate locations that are NaN 
	 * ****************************************************************/
	if(distToLowerIndex!=0 & distToUpperIndex!=0)
	{
		membraneCoordinates[xInd].x = ((double)distToLowerIndex * membraneCoordinates[closestLowerNoneNanIndexLoc[xInd]].x
									+ (double)distToUpperIndex * membraneCoordinates[closestUpperNoneNanIndexLoc[xInd]].x)
									/(double)(distToLowerIndex+distToUpperIndex);
		membraneCoordinates[xInd].y = (distToLowerIndex * membraneCoordinates[closestLowerNoneNanIndexLoc[xInd]].y 
									+ distToUpperIndex * membraneCoordinates[closestUpperNoneNanIndexLoc[xInd]].y)
									/(distToLowerIndex+distToUpperIndex);
		
		membraneNormalVectors[xInd].x = ((double)distToLowerIndex * membraneNormalVectors[closestLowerNoneNanIndexLoc[xInd]].x 
									  + (double)distToUpperIndex * membraneNormalVectors[closestUpperNoneNanIndexLoc[xInd]].x)
									  /(double)(distToLowerIndex+distToUpperIndex);

		membraneNormalVectors[xInd].y = ((double)distToLowerIndex * membraneNormalVectors[closestLowerNoneNanIndexLoc[xInd]].y 
									  + (double)distToUpperIndex * membraneNormalVectors[closestUpperNoneNanIndexLoc[xInd]].y)
									  /(double)(distToLowerIndex+distToUpperIndex);

		double membraneNormalNorm;
		membraneNormalNorm = sqrt( pow(membraneNormalVectors[xInd].x,2) + pow(membraneNormalVectors[xInd].y,2) );
		
		membraneNormalVectors[xInd].x = membraneNormalVectors[xInd].x/membraneNormalNorm;
		membraneNormalVectors[xInd].y = membraneNormalVectors[xInd].y/membraneNormalNorm;
	}
}

/* This function will set trackingFinished to 0 (FALSE),
 * if the euclidian distances between contourCenter and previousContourCenter
 * is >centerTolerance.
 */
__kernel void checkIfCenterConverged(
									__global double2* contourCenter,
									__global double2* previousContourCenter,
									__global int* trackingFinished,
									const double centerTolerance
									)
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	__private double xDistance, yDistance;
	if(xInd==0)
	{
		__private double distance =  length(contourCenter[xInd] - previousContourCenter[xInd]);
		
		if(distance>centerTolerance)
		{
			trackingFinished[0] = 0;
		}
	}
}

__kernel void calculateInterCoordinateAngles(__global double* interCoordinateAngles,
											 __global double2* membraneCoordinates
											 )
{
/* This uses the equation
 * A · B = A B cos θ = |A||B| cos θ
 * to calculate the angles between adjacent coordinates.
 */
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	__private double2 dsVector1, dsVector2;
	
	if(xInd>0 && xInd<xSize-1) // calculate interior gradients
	{
		dsVector1.x = membraneCoordinates[xInd].x - membraneCoordinates[xInd-1].x;
		dsVector1.y = membraneCoordinates[xInd].y - membraneCoordinates[xInd-1].y;
		
		dsVector2.x = membraneCoordinates[xInd+1].x - membraneCoordinates[xInd].x;
		dsVector2.y = membraneCoordinates[xInd+1].y - membraneCoordinates[xInd].y;
	}
	else if(xInd==0) // calculate edge gradient
	{
		dsVector1.x = membraneCoordinates[xInd].x - membraneCoordinates[xSize-1].x;
		dsVector1.y = membraneCoordinates[xInd].y - membraneCoordinates[xSize-1].y;
		
		dsVector2.x = membraneCoordinates[xInd+1].x - membraneCoordinates[xInd].x;
		dsVector2.y = membraneCoordinates[xInd+1].y - membraneCoordinates[xInd].y;
	}
	else if(xInd==xSize-1) // calculate edge gradient
	{
		dsVector1.x = membraneCoordinates[xInd].x - membraneCoordinates[xInd-1].x;
		dsVector1.y = membraneCoordinates[xInd].y - membraneCoordinates[xInd-1].y;
		
		dsVector2.x = membraneCoordinates[0].x - membraneCoordinates[xInd].x;
		dsVector2.y = membraneCoordinates[0].y - membraneCoordinates[xInd].y;
	}

	dsVector1 = normalize(dsVector1);
	dsVector2 = normalize(dsVector2);
	
	//~ double dotProduct = dsVector1.x*dsVector2.x+dsVector1.y*dsVector2.y;
	double dotProduct = dot(dsVector1,dsVector2);
	interCoordinateAngles[xInd] = acos(dotProduct);
	
	if(dotProduct>=1)
	{
		interCoordinateAngles[xInd] = 0;
	}
	if(dotProduct<=-1)
	{
		interCoordinateAngles[xInd] = M_PI;
	}
}

void findClosestGoodCoordinates(const int xInd,
								const int xSize,
								__local int* closestLowerCorrectIndexLoc,
								__local int* closestUpperCorrectIndexLoc,
								__local int* listOfGoodCoordinates,
								int* distToLowerIndex,
								int* distToUpperIndex
								)
{
	__private bool incorrectValueLeft;

	do
	{
		incorrectValueLeft = false;

		if(listOfGoodCoordinates[closestLowerCorrectIndexLoc[xInd]] == 0)
		{
			closestLowerCorrectIndexLoc[xInd] -= 1;
			(*distToLowerIndex)++;
			incorrectValueLeft = true;
		}
		
		if(listOfGoodCoordinates[closestUpperCorrectIndexLoc[xInd]] == 0)
		{
			closestUpperCorrectIndexLoc[xInd] += 1;
			(*distToUpperIndex)++;
			incorrectValueLeft = true;
		}

		if(closestLowerCorrectIndexLoc[xInd]<0) // avoid that we round out array bounds by using periodic boundaries
		{
			closestLowerCorrectIndexLoc[xInd] = closestLowerCorrectIndexLoc[xInd]+xSize;
		}
		if(closestUpperCorrectIndexLoc[xInd]>xSize-1) // avoid that we round out array bounds by using periodic boundaries
		{
			closestUpperCorrectIndexLoc[xInd] = closestUpperCorrectIndexLoc[xInd]-xSize;
		}
	}
	while(incorrectValueLeft);
}

void interpolateIncorrectCoordinates(const int xInd,
									const int xSize,
									__global double2 previousContourCenter[],
									__global double2 membraneCoordinates[],
									__global double2 membraneNormalVectors[],
									__local int* closestLowerCorrectIndexLoc,
									__local int* closestUpperCorrectIndexLoc,
									int* distToLowerIndex,
									int* distToUpperIndex
									)
{
	if(*distToLowerIndex!=0 & *distToUpperIndex!=0)
	{
		membraneCoordinates[xInd].x = ((double)*distToLowerIndex * membraneCoordinates[closestLowerCorrectIndexLoc[xInd]].x 
									+ (double)*distToUpperIndex * membraneCoordinates[closestUpperCorrectIndexLoc[xInd]].x)
									/(double)(*distToLowerIndex+*distToUpperIndex);
		membraneCoordinates[xInd].y = ((double)*distToLowerIndex * membraneCoordinates[closestLowerCorrectIndexLoc[xInd]].y 
									+ (double)*distToUpperIndex * membraneCoordinates[closestUpperCorrectIndexLoc[xInd]].y)
									/(double)(*distToLowerIndex+*distToUpperIndex);
		
		membraneNormalVectors[xInd].x = membraneCoordinates[xInd].x - previousContourCenter[0].x;
		membraneNormalVectors[xInd].y = membraneCoordinates[xInd].y - previousContourCenter[0].y;

		double membraneNormalNorm;
		membraneNormalNorm = sqrt( pow(membraneNormalVectors[xInd].x,2) + pow(membraneNormalVectors[xInd].y,2) );
		
		membraneNormalVectors[xInd].x = membraneNormalVectors[xInd].x/membraneNormalNorm;
		membraneNormalVectors[xInd].y = membraneNormalVectors[xInd].y/membraneNormalNorm;
	}	
}

__kernel void filterJumpedCoordinates(
										__global double2* previousContourCenter,
										__global double2* membraneCoordinates,
										__global double2* membraneNormalVectors,
										__global double2* previousInterpolatedMembraneCoordinates,
										__local int* closestLowerCorrectIndexLoc,
										__local int* closestUpperCorrectIndexLoc,
										__local int* listOfGoodCoordinates,
										const double maxCoordinateShift
										)
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	listOfGoodCoordinates[xInd] = 1;
	closestLowerCorrectIndexLoc[xInd] = xInd;
	closestUpperCorrectIndexLoc[xInd] = xInd;
	__private double distance;
	
	distance =  sqrt(  pow((membraneCoordinates[xInd].x - previousInterpolatedMembraneCoordinates[xInd].x),2)
					 + pow((membraneCoordinates[xInd].y - previousInterpolatedMembraneCoordinates[xInd].y),2)
					 );

	if(distance>maxCoordinateShift)
	{
		listOfGoodCoordinates[xInd] = 0;
	}
	
	__private int distToLowerIndex = 0;
	__private int distToUpperIndex = 0;

	findClosestGoodCoordinates(xInd,
							  xSize,
							  closestLowerCorrectIndexLoc,
							  closestUpperCorrectIndexLoc,
							  listOfGoodCoordinates,
							  &distToLowerIndex,
							  &distToUpperIndex);
							  
	interpolateIncorrectCoordinates(xInd,
									xSize,
									previousContourCenter,
									membraneCoordinates,
									membraneNormalVectors,
									closestLowerCorrectIndexLoc,
									closestUpperCorrectIndexLoc,
									&distToLowerIndex,
									&distToUpperIndex
									);
}

__kernel void filterIncorrectCoordinates(__global double2* previousContourCenter,
										 __global double* interCoordinateAngles,
										 __global double2* membraneCoordinates,
										 __global double2* membraneNormalVectors,
										 __local int* closestLowerCorrectIndexLoc,
										 __local int* closestUpperCorrectIndexLoc,
										 const double maxInterCoordinateAngle
										 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	__private bool incorrectValueLeft;

	closestLowerCorrectIndexLoc[xInd] = xInd;
	closestUpperCorrectIndexLoc[xInd] = xInd;
	
	__private int distToLowerIndex = 0;
	__private int distToUpperIndex = 0;
	do
	{
		incorrectValueLeft = false;
		if(interCoordinateAngles[closestLowerCorrectIndexLoc[xInd]] > maxInterCoordinateAngle)
		{
			closestLowerCorrectIndexLoc[xInd] -= 1;
			distToLowerIndex++;
			incorrectValueLeft = true;
		}
		
		if(interCoordinateAngles[closestUpperCorrectIndexLoc[xInd]] > maxInterCoordinateAngle)
		{
			closestUpperCorrectIndexLoc[xInd] += 1;
			distToUpperIndex++;
			incorrectValueLeft = true;
		}

		if(closestLowerCorrectIndexLoc[xInd]<0) // avoid that we round out array bounds by using periodic boundaries
		{
			closestLowerCorrectIndexLoc[xInd] = closestLowerCorrectIndexLoc[xInd]+xSize;
		}
		if(closestUpperCorrectIndexLoc[xInd]>xSize-1) // avoid that we round out array bounds by using periodic boundaries
		{
			closestUpperCorrectIndexLoc[xInd] = closestUpperCorrectIndexLoc[xInd]-xSize;
		}
	}
	while(incorrectValueLeft);
	
	/* *****************************************************************
	 * interpolate locations that are NaN 
	 * ****************************************************************/
	if(distToLowerIndex!=0 & distToUpperIndex!=0)
	{
		membraneCoordinates[xInd].x = ((double)distToLowerIndex * membraneCoordinates[closestLowerCorrectIndexLoc[xInd]].x 
									+ (double)distToUpperIndex * membraneCoordinates[closestUpperCorrectIndexLoc[xInd]].x)
									/(double)(distToLowerIndex+distToUpperIndex);
		membraneCoordinates[xInd].y = ((double)distToLowerIndex * membraneCoordinates[closestLowerCorrectIndexLoc[xInd]].y 
									+ (double)distToUpperIndex * membraneCoordinates[closestUpperCorrectIndexLoc[xInd]].y)
									/(double)(distToLowerIndex+distToUpperIndex);

		membraneNormalVectors[xInd].x = membraneCoordinates[xInd].x - previousContourCenter[0].x;
		membraneNormalVectors[xInd].y = membraneCoordinates[xInd].y - previousContourCenter[0].y;
		
		double membraneNormalNorm;
		membraneNormalNorm = sqrt( pow(membraneNormalVectors[xInd].x,2) + pow(membraneNormalVectors[xInd].y,2) );

		membraneNormalVectors[xInd].x = membraneNormalVectors[xInd].x/membraneNormalNorm;
		membraneNormalVectors[xInd].y = membraneNormalVectors[xInd].y/membraneNormalNorm;
	}
}

__kernel void interpolatePolarCoordinatesLinear(__global double2* membranePolarCoordinates,
											  __global double2* radialVectors,
											  __global double2* contourCenter,
											  __global double2* membraneCoordinates,
											  __global double2* interpolatedMembraneCoordinates,
											  __global double* interpolationAngles,
											  const int nrOfAnglesToCompare
											 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	__private int index;
	__private int lowerIndex;
	__private int upperIndex;
	__private double lowerAngle;
	__private double upperAngle;
	__private double interpolationAngle;
	
	// radialLineDirectionVector <-> b
	// radialLineBasePoint <-> c
	__private double2 radialLineBasePoint = contourCenter[0];
	__private double2 radialLineDirectionVector = radialVectors[xInd];
	
	__private double distanceFromCenter = 160000;
	__private double2 interpolatedMembranePoint;
	__private double2 radialVector;
		
	for(index=xInd-nrOfAnglesToCompare/2+1;index<xInd+nrOfAnglesToCompare/2;index++)
	{
		lowerIndex = index;
		upperIndex = index+1;
		
		///* Check index sanity */
		if(lowerIndex<0){
			lowerIndex = lowerIndex + xSize;
		}
		if(lowerIndex>xSize-1){
			lowerIndex = lowerIndex - xSize;
		}

		if(upperIndex<0){
			upperIndex = upperIndex + xSize;
		}
		if(upperIndex>xSize-1){
			upperIndex = upperIndex - xSize;
		}
		
		lowerAngle = membranePolarCoordinates[lowerIndex].x;
		upperAngle = membranePolarCoordinates[upperIndex].x;
		interpolationAngle = interpolationAngles[xInd];
		
		if(lowerAngle>upperAngle){
			lowerAngle = lowerAngle - 2 * M_PI;
			if(interpolationAngle>0){
				interpolationAngle = interpolationAngle - 2 * M_PI;
			}
		}
		
		// take into account the case, where contour line-segment coincides with the radial line
		//~ if( interpolationAngles[xInd] == lowerAngle && interpolationAngles[xInd] == upperAngle )
		if( interpolationAngle == lowerAngle && interpolationAngle == upperAngle )
		{
			radialVector = membraneCoordinates[lowerIndex] - radialLineBasePoint;
			if(length(radialVector)<distanceFromCenter){
				distanceFromCenter = length(radialVector);
				interpolatedMembranePoint = membraneCoordinates[lowerIndex];
			}
			radialVector = membraneCoordinates[upperIndex] - radialLineBasePoint;
			if(length(radialVector)<distanceFromCenter){
				distanceFromCenter = length(radialVector);
				interpolatedMembranePoint = membraneCoordinates[upperIndex];
			}
			break;
		}
		
		//~ if( membranePolarTheta[xInd] >= interpolationAngles[lowerIndex] && membranePolarTheta[xInd] < interpolationAngles[upperIndex] )
		if( interpolationAngle >= lowerAngle && interpolationAngle < upperAngle )
		{
			// lineSegmentDirectionVector <-> e
			// lineSegmentBasePoint <-> f
			__private double2 lineSegmentDirectionVector;
			__private double2 lineSegmentBasePoint;
			
			// calculate paramaters of the straight, that passes through contour-segment
			lineSegmentDirectionVector = membraneCoordinates[upperIndex] - membraneCoordinates[lowerIndex];
			lineSegmentDirectionVector = normalize(lineSegmentDirectionVector);
						
			lineSegmentBasePoint = membraneCoordinates[lowerIndex];
			
			// check if contour line-segment is parallel to the radial line
			
			// calculate intercept point between radial line and the line-segment of the contour
			// radialLineDirectionVector <-> b
			// radialLineBasePoint <-> c
			// lineSegmentDirectionVector <-> e
			// lineSegmentBasePoint <-> f
			double m;
			m = ( lineSegmentDirectionVector.y*(lineSegmentBasePoint.x-radialLineBasePoint.x)
			     -lineSegmentDirectionVector.x*(lineSegmentBasePoint.y-radialLineBasePoint.y) )
			    / ( radialLineDirectionVector.x*lineSegmentDirectionVector.y
			       -radialLineDirectionVector.y*lineSegmentDirectionVector.x );
			
			__private double2 interpolatedMembranePointTMP;
			interpolatedMembranePointTMP = m * radialLineDirectionVector + radialLineBasePoint;
			
			radialVector = interpolatedMembranePointTMP - radialLineBasePoint;
			if(length(radialVector)<distanceFromCenter){
				distanceFromCenter = length(radialVector);
				interpolatedMembranePoint = interpolatedMembranePointTMP;
			}
			break;
		}

	}
	interpolatedMembraneCoordinates[xInd] = interpolatedMembranePoint;
}
