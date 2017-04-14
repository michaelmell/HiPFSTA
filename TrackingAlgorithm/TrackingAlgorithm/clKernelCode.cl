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

__kernel void calculateMembraneNormalVectors(__global double2* membraneCoordinates,
											 __global double2* membraneNormalVectors
											)
	{
		const int xInd = get_global_id(0);
		const int xSize = get_global_size(0);
		
		__private double vectorNorm;
		
		// NOTE: we use bilinear interpolation to calculate the gradient vectors
		if(xInd>0 && xInd<xSize-1){
			membraneNormalVectors[xInd].y = -(  (membraneCoordinates[xInd].x - membraneCoordinates[xInd-1].x)
			                                  + (membraneCoordinates[xInd+1].x - membraneCoordinates[xInd].x) )/2;
			membraneNormalVectors[xInd].x =  (  (membraneCoordinates[xInd].y - membraneCoordinates[xInd-1].y)
			                                  + (membraneCoordinates[xInd+1].y - membraneCoordinates[xInd].y) )/2;
		}
		else if(xInd==0){
			membraneNormalVectors[xInd].y = -(  (membraneCoordinates[xInd].x - membraneCoordinates[xSize-1].x)
			                                  + (membraneCoordinates[xInd+1].x - membraneCoordinates[xInd].x) )/2;
			membraneNormalVectors[xInd].x =  (  (membraneCoordinates[xInd].y - membraneCoordinates[xSize-1].y)
			                                  + (membraneCoordinates[xInd+1].y - membraneCoordinates[xInd].y) )/2;
		}
		else if(xInd==xSize-1){
			membraneNormalVectors[xInd].y = -(  (membraneCoordinates[xInd].x - membraneCoordinates[xInd-1].x)
			                                  + (membraneCoordinates[0].x - membraneCoordinates[xInd].x) )/2;
			membraneNormalVectors[xInd].x =  (  (membraneCoordinates[xInd].y - membraneCoordinates[xInd-1].y)
			                                  + (membraneCoordinates[0].y - membraneCoordinates[xInd].y) )/2;
		}
		
		barrier(CLK_GLOBAL_MEM_FENCE);
		vectorNorm = sqrt(pow(membraneNormalVectors[xInd].x,2) + pow(membraneNormalVectors[xInd].y,2));
		
		membraneNormalVectors[xInd].x = membraneNormalVectors[xInd].x/vectorNorm;
		membraneNormalVectors[xInd].y = membraneNormalVectors[xInd].y/vectorNorm;
	}

__kernel void calculateDs(
						 __global double2* membraneCoordinates,
						 __global double* ds
						 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	if(xInd>=1){
		ds[xInd] = sqrt(pow((membraneCoordinates[xInd].x - membraneCoordinates[xInd-1].x),2)
				  + 	pow((membraneCoordinates[xInd].y - membraneCoordinates[xInd-1].y),2)
				  );
	}
	else if(xInd==0){
		ds[xInd] = sqrt(pow((membraneCoordinates[xInd].x - membraneCoordinates[xSize-1].x),2)
				  + 	pow((membraneCoordinates[xInd].y - membraneCoordinates[xSize-1].y),2)
				  );
	}
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

__kernel void cart2pol(__global double* membraneCoordinatesX,
					   __global double* membraneCoordinatesY,
					   __global double* membranePolarRadius,
					   __global double* membranePolarTheta,
					   __global double2* contourCenter
					  )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	double2 contourCenterLoc;
	contourCenterLoc = contourCenter[0];
	
	membranePolarTheta[xInd] =  atan2( membraneCoordinatesY[xInd] - contourCenterLoc.y,
									   membraneCoordinatesX[xInd] - contourCenterLoc.x);
	membranePolarRadius[xInd] =  sqrt( pow((membraneCoordinatesY[xInd] - contourCenterLoc.y),2)
									 + pow((membraneCoordinatesX[xInd] - contourCenterLoc.x),2) 
									 );
}

__kernel void pol2cart(__global double* membraneCoordinatesX,
					   __global double* membraneCoordinatesY,
					   __global double* membranePolarRadius,
					   __global double* membranePolarTheta,
					   __global double2* contourCenter
					  )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	double2 contourCenterLoc;
	contourCenterLoc = contourCenter[0];
	
	membraneCoordinatesX[xInd] = contourCenterLoc.x + membranePolarRadius[xInd] * cos(membranePolarTheta[xInd]);
	membraneCoordinatesY[xInd] = contourCenterLoc.y + membranePolarRadius[xInd] * sin(membranePolarTheta[xInd]);
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


__kernel void setIterationFinished(__global int* iterationFinished) // will set self.dev_iterationFinished to true, when called
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	//~ printf("iterationFinished before setting to true: %d\n",iterationFinished[0]);
	iterationFinished[0] = 1;
	//~ printf("iterationFinished: %d\n",iterationFinished[0]);
}

__kernel void checkIfTrackingFinished(
									__global double* interpolatedMembraneCoordinatesX,
									__global double* interpolatedMembraneCoordinatesY,
									__global double* previousInterpolatedMembraneCoordinatesX,
									__global double* previousInterpolatedMembraneCoordinatesY,
									__global int* trackingFinished,
									const double coordinateTolerance
									) // will set self.dev_iterationFinished to true, when called
{
	// this function will set trackingFinished to 0 (FALSE), if the euclidian distance between any coordinate of the interpolated contours is >coordinateTolerance
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	__private double distance;
	distance =  sqrt(  pow((interpolatedMembraneCoordinatesX[xInd] - previousInterpolatedMembraneCoordinatesX[xInd]),2)
					 + pow((interpolatedMembraneCoordinatesY[xInd] - previousInterpolatedMembraneCoordinatesY[xInd]),2)
					 );
	
	//~ if(xInd==100){
		//~ printf("coordinateTolerance: %f\n",coordinateTolerance);
		//~ printf("distance: %f\n",distance);
	//~ }
	
	if(distance>coordinateTolerance){
		trackingFinished[0] = 0;
		//~ printf("xInd: %d\n",xInd);
		//~ printf("distance: %f\n",distance);
	}
	
	//~ printf("iterationFinished before setting to true: %d\n",iterationFinished[0]);
	//~ printf("iterationFinished: %d\n",iterationFinished[0]);
}

__kernel void sortCoordinates(__global double* membranePolarRadius,
							  __global double* membranePolarTheta,
							  __global double* membraneCoordinatesX,
							  __global double* membraneCoordinatesY,
							  __global double* membraneNormalVectorsX,
							  __global double* membraneNormalVectorsY,
							  //~ __local double* membranePolarRadiusLoc,
							  //~ __local double* membranePolarThetaLoc,
							  //~ __local double* membranePolarRadiusLoc,
							  //~ __local double* membranePolarThetaLoc,
							  const int nrOfContourPoints
								  )
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
{
	const int xInd = get_global_id(1);
	const int yInd = get_global_id(0);
	const int xSize = get_global_size(1);
	const int ySize = get_global_size(0);

	if(xInd==0 && yInd==0){
	//~ procedure bubbleSort( A : list of sortable items )
		//~ n = length(A)
		int n = nrOfContourPoints;
		//~ repeat
		while(n!=0){
		   //~ newn = 0
		   int newn = 0;
		   //~ for i = 1 to n-1 inclusive do
		   for(int i=1;i<=n-1;i++){
			  //~ if A[i-1] > A[i] then
				if(membranePolarTheta[i-1]>membranePolarTheta[i]){
				 //~ swap(A[i-1], A[i])
					__private double thetaTmp = membranePolarTheta[i-1];
					membranePolarTheta[i-1] = membranePolarTheta[i];
					membranePolarTheta[i] = thetaTmp;
					
					__private double radiusTmp = membranePolarRadius[i-1];
					membranePolarRadius[i-1] = membranePolarRadius[i];
					membranePolarRadius[i] = radiusTmp;
					
					__private double xCoordTmp = membraneCoordinatesX[i-1];
					membraneCoordinatesX[i-1] = membraneCoordinatesX[i];
					membraneCoordinatesX[i] = xCoordTmp;

					__private double yCoordTmp = membraneCoordinatesY[i-1];
					membraneCoordinatesY[i-1] = membraneCoordinatesY[i];
					membraneCoordinatesY[i] = yCoordTmp;

					__private double xNormalTmp = membraneNormalVectorsX[i-1];
					membraneNormalVectorsX[i-1] = membraneNormalVectorsX[i];
					membraneNormalVectorsX[i] = xNormalTmp;

					__private double yNormalTmp = membraneNormalVectorsY[i-1];
					membraneNormalVectorsY[i-1] = membraneNormalVectorsY[i];
					membraneNormalVectorsY[i] = yNormalTmp;
				 //~ newn = i
				 newn = i;
			  //~ end if
				}
		   //~ end for
		   }
		   //~ n = newn
		   n = newn;
	   }
		//~ until n = 0
	//~ end procedure
	}

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
									   //~ __global double* fitIntercept,
									   //~ __global double* fitIncline,
									   const int meanParameter,
									   __constant double* meanRangeXvalues, // this should be local or constant
									   const double meanRangePositionOffset,
									   __local double* localMembranePositionsX,
									   __local double* localMembranePositionsY,
									   //~ __global double* localMembranePositionsX,
									   //~ __global double* localMembranePositionsY,
									   __global double* membraneCoordinatesX,
									   __global double* membraneCoordinatesY,
									   __global double* membraneNormalVectorsX,
									   __global double* membraneNormalVectorsY,
									   __global double* fitInclines,
									   const int coordinateStartingIndex,
									   const double inclineTolerance)
{
	// self.trackingWorkGroupSize, self.trackingGlobalSize
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
	
	// Notes:
	// 1) xInd = xGroupSize*xSizeLoc+xIndLoc
	
	//~ if(xInd==0 && yInd==0){
		//~ printf("xSize: %d\n",xSize);
		//~ printf("ySize: %d\n",ySize);
		//~ printf("xSizeLoc: %d\n",xSizeLoc);
		//~ printf("ySizeLoc: %d\n",ySizeLoc);
		//~ printf("xGroupId: %d\n",xGroupId);
		//~ printf("yGroupId: %d\n",yGroupId);
		//~ printf("coordinateStartingIndex: %d\n",coordinateStartingIndex);
	//~ }
	
	//~ const int coordinateIndex = coordinateStartingIndex;
	const int coordinateIndex = coordinateStartingIndex + yGroupId*ySizeLoc + yIndLoc;
	
	//~ if(xInd==10){
		//~ printf("yIndLoc: %d\n",yIndLoc);
		//~ printf("xIndLoc: %d\n",xIndLoc);
		//~ printf("yGroupId: %d\n",yGroupId);
		//~ printf("ySizeLoc: %d\n",ySizeLoc);
		//~ printf("coordinateIndex: %d\n",coordinateIndex);
	//~ }
	
	__private double lineIntensities[400];
	
	__private double2 membraneNormalVector;
	membraneNormalVector.x = membraneNormalVectorsX[coordinateIndex];
	membraneNormalVector.y = membraneNormalVectorsY[coordinateIndex];
	
	// matrix multiplication with linear array of sequential 2x2 rotation matrices
	//~ __private double2 rotatedUnitVector;
	//~ rotatedUnitVector.x = localRotationMatrices[4*xIndLoc+0] * membraneNormalVector.x
	                     //~ + localRotationMatrices[4*xIndLoc+1] * membraneNormalVector.y;
	//~ rotatedUnitVector.y = localRotationMatrices[4*xIndLoc+2] * membraneNormalVector.x
	                     //~ + localRotationMatrices[4*xIndLoc+3] * membraneNormalVector.y;
	rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].x = localRotationMatrices[4*xIndLoc+0] * membraneNormalVector.x
												   + localRotationMatrices[4*xIndLoc+1] * membraneNormalVector.y;
	rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].y = localRotationMatrices[4*xIndLoc+2] * membraneNormalVector.x
												   + localRotationMatrices[4*xIndLoc+3] * membraneNormalVector.y;
	
	//if(xInd==10 && yInd==10){
		//printf("localRotationMatrices[4*xIndLoc+0]: %f\n",localRotationMatrices[4*xIndLoc+0]);
		//printf("localRotationMatrices[4*xIndLoc+1]: %f\n",localRotationMatrices[4*xIndLoc+1]);
		//printf("localRotationMatrices[4*xIndLoc+2]: %f\n",localRotationMatrices[4*xIndLoc+2]);
		//printf("localRotationMatrices[4*xIndLoc+3]: %f\n",localRotationMatrices[4*xIndLoc+3]);
		////~ printf("localRotationMatrices[4*xInd+0]: %f\n",localRotationMatrices[4*xInd+0]);
		////~ printf("localRotationMatrices[4*xInd+1]: %f\n",localRotationMatrices[4*xInd+1]);
		////~ printf("localRotationMatrices[4*xInd+2]: %f\n",localRotationMatrices[4*xInd+2]);
		////~ printf("localRotationMatrices[4*xInd+3]: %f\n",localRotationMatrices[4*xInd+3]);
	//}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	__private int maxIndex;
	__private int minIndex;
	__private double minValue = 32000; // initialize value with first value of array
	__private double maxValue = 0; // initialize value with first value of array
	
	__private double2 Coords;
	__private double2 NormCoords;
	__private const int2 dims = get_image_dim(Img);
	
	__private double2 basePoint = {membraneCoordinatesX[coordinateIndex],membraneCoordinatesY[coordinateIndex]};

	//~ if(xInd==0 && yInd==0){
		//~ printf("basePoint.x: %f\n",basePoint.x);
		//~ printf("basePoint.y: %f\n",basePoint.y);
	//~ }
	
	//if(xInd==0 && yInd==0){
		//printf("yIndLoc: %d\n",yIndLoc);
		//printf("xIndLoc: %d\n",xIndLoc);
		//printf("yGroupId: %d\n",yGroupId);
		//printf("ySizeLoc: %d\n",ySizeLoc);
		//printf("coordinateIndex: %d\n",coordinateIndex);
		//printf("membraneNormalVector.x: %f\n",membraneNormalVector.x);
		//printf("membraneNormalVector.y: %f\n",membraneNormalVector.y);
		//printf("basePoint.x: %f\n",basePoint.x);
		//printf("basePoint.y: %f\n",basePoint.y);
	//}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int index=0;index<imgSizeY;index++){
		//~ Coords.x = basePoint.x + rotatedUnitVector.x * linFitSearchRangeXvalues[index];
		//~ Coords.y = basePoint.y + rotatedUnitVector.y * linFitSearchRangeXvalues[index];
		Coords.x = basePoint.x + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].x * linFitSearchRangeXvalues[index];
		Coords.y = basePoint.y + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].y * linFitSearchRangeXvalues[index];
		
		NormCoords = Coords/convert_double2(dims);
		
		float2 fNormCoords;
		fNormCoords.x = (float)NormCoords.x;
		fNormCoords.y = (float)NormCoords.y;
		
		lineIntensities[index] = read_imagef(Img, sampler, fNormCoords).x;
		//~ interpolatedIntensities[xInd+index*imgSizeX] = read_imagef(Img, sampler, NormCoords).x;
		
		maxIndex = select(maxIndex,index,(maxValue < lineIntensities[index]));
		maxValue = select(maxValue,lineIntensities[index],(long)(maxValue < lineIntensities[index]));
		//~ if(maxValue < lineIntensities[index]){
			//~ maxIndex = index;
			//~ maxValue = lineIntensities[index];
		//~ }
		minIndex = select(minIndex,index,(minValue > lineIntensities[index]));
		minValue = select(minValue,lineIntensities[index],(long)(minValue > lineIntensities[index]));
		//~ if(minValue > lineIntensities[index]){
			//~ minIndex = index;
			//~ minValue = lineIntensities[index];
		//~ }
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	//~ __private int gradientCenterIndex, tmp;
	__private int gradientCenterIndex;
	__private double gradientCenterValue = minValue+(maxValue-minValue)/2.0;
	__private double minValue2 = 20000;
	__private double refValue;
	
	__private double a=0.0, b=0.0, siga=0.0, sigb=0.0, chi2=0.0;
		
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	if(minIndex<maxIndex){
		for(int index=minIndex;index<maxIndex;index++){
			refValue = fabs(lineIntensities[index]-gradientCenterValue);
			gradientCenterIndex = select(gradientCenterIndex,index,(minValue2 > refValue));
			minValue2 = select(minValue2,refValue,(long)(minValue2 > refValue));
		}
		// "r = a > b ? a : b" corresponds to: "r = select(b, a, a > b)", corresponds to "if(a>b){r=a}; else{r=b}"
		// reference: http://stackoverflow.com/questions/7635706/opencl-built-in-function-select
		
		linearFit(linFitSearchRangeXvalues, lineIntensities, gradientCenterIndex, linFitParameter, &a, &b, &siga, &sigb, &chi2);
		//~ fitIntercept[xIndLoc] = a;
		fitIntercept[xIndLoc+yIndLoc*xSizeLoc] = a;
		//~ fitIncline[xIndLoc] = b;
		fitIncline[xIndLoc+yIndLoc*xSizeLoc] = b;
	}
	else{
	// ToDo:
	// else:
		fitIntercept[xIndLoc+yIndLoc*xSizeLoc] = 0; // so that later they are not counted in the weighted sum (see below...)
		fitIncline[xIndLoc+yIndLoc*xSizeLoc] = 0;
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	__private double meanIntensity = 0.0;
	for(int index=0;index<meanParameter;index++){
		//~ Coords.x = basePoint.x + rotatedUnitVector.x * ( meanRangeXvalues[index] + meanRangePositionOffset );
		//~ Coords.y = basePoint.y + rotatedUnitVector.y * ( meanRangeXvalues[index] + meanRangePositionOffset );
		Coords.x = basePoint.x + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].x * ( meanRangeXvalues[index] + meanRangePositionOffset );
		Coords.y = basePoint.y + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].y * ( meanRangeXvalues[index] + meanRangePositionOffset );
		
		NormCoords = Coords/convert_double2(dims);
		
		float2 fNormCoords;
		fNormCoords.x = (float)NormCoords.x;
		fNormCoords.y = (float)NormCoords.y;

		meanIntensity = meanIntensity + read_imagef(Img, sampler, fNormCoords).x;
	}
	meanIntensity = meanIntensity/convert_float(meanParameter);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	__private double relativeMembranePositionLocalCoordSys;
	//~ __private double membranePositionsX, membranePositionsY;
	//~ relativeMembranePositionLocalCoordSys = (meanIntensity-fitIntercept[xIndLoc])/fitIncline[xIndLoc];
	if(fitIncline[xIndLoc+yIndLoc*xSizeLoc] != 0){
		relativeMembranePositionLocalCoordSys = (meanIntensity-fitIntercept[xIndLoc+yIndLoc*xSizeLoc])/fitIncline[xIndLoc+yIndLoc*xSizeLoc];
	}
	else{
		relativeMembranePositionLocalCoordSys = 0;
	}
	
	//~ if(fitIncline[xIndLoc+yIndLoc*xSizeLoc]  == 0){
		//~ printf("A fitIncline at coordinateIndex %d is 0.\n",coordinateIndex);
	//~ }
	//~ 
	//~ if(coordinateIndex  == 2029){
		//~ printf("fitIncline at coordinateIndex %d: %f\n",coordinateIndex,fitIncline[xIndLoc+yIndLoc*xSizeLoc]);
	//~ }
	
	//~ localMembranePositionsX[xIndLoc] = basePoint.x + rotatedUnitVector.x * relativeMembranePositionLocalCoordSys;
	//~ localMembranePositionsY[xIndLoc] = basePoint.y + rotatedUnitVector.y * relativeMembranePositionLocalCoordSys;
	//~ localMembranePositionsX[xIndLoc+yIndLoc*xSizeLoc] = basePoint.x + rotatedUnitVector.x * relativeMembranePositionLocalCoordSys;
	//~ localMembranePositionsY[xIndLoc+yIndLoc*xSizeLoc] = basePoint.y + rotatedUnitVector.y * relativeMembranePositionLocalCoordSys;
	localMembranePositionsX[xIndLoc+yIndLoc*xSizeLoc] = basePoint.x + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].x * relativeMembranePositionLocalCoordSys;
	localMembranePositionsY[xIndLoc+yIndLoc*xSizeLoc] = basePoint.y + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].y * relativeMembranePositionLocalCoordSys;
	
	//~ if(xInd==0 && yIndLoc*xSizeLoc<64){
		//~ printf("localMembranePositionsX[xInd]: %f\n",localMembranePositionsX[xIndLoc+yIndLoc*xSizeLoc]);
		//~ printf("localMembranePositionsY[xInd]: %f\n",localMembranePositionsY[xIndLoc+yIndLoc*xSizeLoc]);
	//~ }
	//~ if(xInd==0 && coordinateIndex==200){
		//~ printf("localMembranePositionsX[xInd]: %f\n",localMembranePositionsX[xInd]);
		//~ printf("localMembranePositionsY[xInd]: %f\n",localMembranePositionsY[xInd]);
	//~ }
	
	write_mem_fence(CLK_LOCAL_MEM_FENCE);

	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	//if(xIndLoc==10 && yInd==1){
		//printf("basePoint.x: %f\n",basePoint.x);
		//printf("basePoint.y: %f\n",basePoint.y);
		//printf("localMembranePositionsX[0]: %f\n",localMembranePositionsX[0]);
		//printf("localMembranePositionsX[1]: %f\n",localMembranePositionsX[1]);
		//printf("localMembranePositionsX[2]: %f\n",localMembranePositionsX[2]);
		//printf("localMembranePositionsX[3]: %f\n",localMembranePositionsX[3]);
	//}
	
	//~ if(xInd==0 && yInd==0){
		//~ printf("xSize: %d\n",xSize);
		//~ printf("ySize: %d\n",ySize);
		//~ printf("xSizeLoc: %d\n",xSizeLoc);
		//~ printf("ySizeLoc: %d\n",ySizeLoc);
		//~ printf("xGroupId: %d\n",xGroupId);
		//~ printf("yGroupId: %d\n",yGroupId);
		//~ printf("coordinateStartingIndex: %d\n",coordinateStartingIndex);
	//~ }
	
	/* *****************************************************************
	 * Find largest inclination value in workgroup and save to maxFitIncline
	 * ****************************************************************/
	__local double maxFitIncline;
	//~ __local double minIncline;
	//~ if(coordinateIndex==829&xIndLoc==0){
	if(xIndLoc==0&yIndLoc==0){
		maxFitIncline = 0.0;
		//~ minIncline = 100.0; // TODO: this is only for debugging; remove once done
		for(int index=0;index<xSizeLoc*ySizeLoc;index++){
			if(fitIncline[index]>maxFitIncline){
				maxFitIncline = fitIncline[index];
			}
			//~ if(fitIncline[index]<minIncline){ // TODO: this is only for debugging; remove once done
				//~ minIncline = fitIncline[index];
			//~ }
		}
	}

	//~ __private double inclineTolerance = 0.7; // TODO: this should be made an algorithm parameter in the settings file
	//~ __private double inclineTolerance = -0.9; // TODO: this should be made an algorithm parameter in the settings file

	//if(xIndLoc==0&coordinateIndex>1850&coordinateIndex<1855){
		////~ printf("xSize: %d\n",xSize);
		////~ printf("ySize: %d\n",ySize);
		////~ printf("xSizeLoc: %d\n",xSizeLoc);
		////~ printf("ySizeLoc: %d\n",ySizeLoc);
		//printf("maxFitIncline: %f\n",maxFitIncline);
		//printf("minIncline: %f\n",minIncline);
		//printf("inclineTolerance*maxFitIncline: %f\n",inclineTolerance*maxFitIncline);
		//printf("inclineTolerance: %f\n",inclineTolerance);
	//}
	
	write_mem_fence(CLK_LOCAL_MEM_FENCE);

	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	
	__private double xTmp = 0.0, yTmp = 0.0, inclineSum = 0.0;
	//~ __private double xMembraneNormalTmp = 0.0, yMembraneNormalTmp = 0.0, membraneNormalNorm, membraneNormalNormNew, membraneNormalNorm2;
	__private double xMembraneNormalTmp = 0.0, yMembraneNormalTmp = 0.0, membraneNormalNorm;
	//~ [xIndLoc+yIndLoc*xSizeLoc]
	
	if(xIndLoc==0){
		//~ for(int yIndLoc=get_local_id(0);yIndLoc++;yIndLoc<ySizeLoc){
			//~ for(int index=get_local_id(1);index<xSize;index++){
			for(int index=0;index<xSize;index++){
				if(fitIncline[index+yIndLoc*xSizeLoc]>inclineTolerance*maxFitIncline){
					//~ xTmp += fitIncline[index] * localMembranePositionsX[index];
					//~ yTmp += fitIncline[index] * localMembranePositionsY[index];
					xTmp += fitIncline[index+yIndLoc*xSizeLoc] * localMembranePositionsX[index+yIndLoc*xSizeLoc];
					yTmp += fitIncline[index+yIndLoc*xSizeLoc] * localMembranePositionsY[index+yIndLoc*xSizeLoc];
					
					xMembraneNormalTmp += fitIncline[index+yIndLoc*xSizeLoc] * rotatedUnitVector2[index+yIndLoc*xSizeLoc].x;
					yMembraneNormalTmp += fitIncline[index+yIndLoc*xSizeLoc] * rotatedUnitVector2[index+yIndLoc*xSizeLoc].y;
					
					//~ inclineSum += fitIncline[index];
					inclineSum += fitIncline[index+yIndLoc*xSizeLoc];
				}
				//~ if(coordinateIndex==829){
					//~ printf("fitIncline: %f\n",fitIncline[index+yIndLoc*xSizeLoc]);
				//~ }
				

			//~ if(coordinateIndex==2000){
				//~ printf("fitIncline: %f\n",fitIncline[index+yIndLoc*xSizeLoc]);
			//~ }
			
			}
			membraneCoordinatesX[coordinateIndex] = xTmp/inclineSum;
			membraneCoordinatesY[coordinateIndex] = yTmp/inclineSum;
			fitInclines[coordinateIndex] = maxFitIncline;
			
			xMembraneNormalTmp = xMembraneNormalTmp/inclineSum;
			yMembraneNormalTmp = yMembraneNormalTmp/inclineSum;

			//~ if(coordinateIndex==2000){
				//~ printf("inclineSum: %f\n",inclineSum);
			//~ }
			
			membraneNormalNorm = sqrt( pow(xMembraneNormalTmp,2) + pow(yMembraneNormalTmp,2) );
			
			//~ if(coordinateIndex==2000){
				//~ printf("membraneNormalNorm: %f\n",membraneNormalNorm);
			//~ }

			xMembraneNormalTmp = xMembraneNormalTmp/membraneNormalNorm;
			yMembraneNormalTmp = yMembraneNormalTmp/membraneNormalNorm;
			
			membraneNormalVectorsX[coordinateIndex] = xMembraneNormalTmp;
			membraneNormalVectorsY[coordinateIndex] = yMembraneNormalTmp;

			//~ if(membraneCoordinatesX[coordinateIndex]>300 ||
			   //~ membraneCoordinatesX[coordinateIndex]<0 ||
			   //~ membraneCoordinatesY[coordinateIndex]>300 ||
			   //~ membraneCoordinatesY[coordinateIndex]<0 ){
			   //~ printf("xInd: %f\n",xInd);
			//~ }

			//~ if(coordinateIndex==2000){
				//~ printf("xMembraneNormalTmp: %f\n",xMembraneNormalTmp);
				//~ printf("yMembraneNormalTmp: %f\n",yMembraneNormalTmp);
			//~ }

			
			//~ membraneNormalNormNew = sqrt( pow(xMembraneNormalTmp,2) + pow(yMembraneNormalTmp,2) );
			
			//~ membraneNormalNorm2 = sqrt( pow(membraneNormalVectorsX[coordinateIndex],2) + pow(membraneNormalVectorsY[coordinateIndex],2) );
			
			//~ if(yIndLoc==0 && coordinateIndex == 1024){
				//~ printf("membraneNormalVectorsX[coordinateIndex]: %f\n",membraneNormalVectorsX[coordinateIndex]);
				//~ printf("membraneNormalVectorsY[coordinateIndex]: %f\n",membraneNormalVectorsY[coordinateIndex]);
				//~ printf("membraneNormalNorm2: %f\n",membraneNormalNorm2);
				//~ printf("xMembraneNormalTmp: %f\n",xMembraneNormalTmp);
				//~ printf("yMembraneNormalTmp: %f\n",yMembraneNormalTmp);
				//~ printf("membraneNormalNorm: %f\n",membraneNormalNorm);
				//~ printf("membraneNormalNormNew: %f\n",membraneNormalNormNew);
			//~ }
			
			//~ printf("coordinateIndex: %d\n",coordinateIndex);
			//~ printf("membraneCoordinatesX[coordinateIndex]: %f\n",membraneCoordinatesX[coordinateIndex]);

		//~ }
	}
	
	//~ if(xInd==0 && yIndLoc==0){
		//~ for(int index=0;index<xSize;index++){
			//~ printf("localMembranePositionsX[xInd]: %f\n",localMembranePositionsX[index]);
			//~ printf("localMembranePositionsY[xInd]: %f\n",localMembranePositionsY[index]);
		//~ }
		//~ printf("xSize: %d\n",xSize);
		//~ printf("xSizeLoc: %d\n",xSizeLoc);
	//~ }
}

__kernel void filterNanValues(__global double* membraneCoordinatesX,
							  __global double* membraneCoordinatesY,
							  __global double* membraneNormalVectorsX,
							  __global double* membraneNormalVectorsY,
							  __local int* closestLowerNoneNanIndexLoc,
							  __local int* closestUpperNoneNanIndexLoc
							  //~ __global double* dbgOut,
							  //~ __global double* dbgOut2
							 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	__private bool NanValueLeft;
	//~ __local int closestLowerNoneNanIndexLoc[2048];
	//~ __local int closestUpperNoneNanIndexLoc[2048];
	
	//~ closestLowerNoneNanIndexLoc[xInd] = closestLowerNoneNanIndex[xInd];
	//~ closestUpperNoneNanIndexLoc[xInd] = closestUpperNoneNanIndex[xInd];
	
	closestLowerNoneNanIndexLoc[xInd] = xInd;
	closestUpperNoneNanIndexLoc[xInd] = xInd;

	//~ if(xInd>1998&xInd<2002){
		//~ printf("closestLowerNoneNanIndex[xInd], before: %d\n",closestLowerNoneNanIndexLoc[xInd]);
		//~ printf("closestUpperNoneNanIndex[xInd], before: %d\n",closestUpperNoneNanIndexLoc[xInd]);
	//~ }
	
	__private int distToLowerIndex = 0;
	__private int distToUpperIndex = 0;
	do{
		NanValueLeft = false;
		if(isnan(membraneCoordinatesX[closestLowerNoneNanIndexLoc[xInd]])){
			closestLowerNoneNanIndexLoc[xInd] -= 1;
			distToLowerIndex++;
			NanValueLeft = true;
		}

		if(isnan(membraneCoordinatesX[closestUpperNoneNanIndexLoc[xInd]])){
			closestUpperNoneNanIndexLoc[xInd] += 1;
			distToUpperIndex++;
			NanValueLeft = true;
		}
		//~ id = getGlobalId();
		//~ output[id] = input[id] * input[id];
		if(closestLowerNoneNanIndexLoc[xInd]<0){ // avoid that we round out array bounds by using periodic boundaries
			closestLowerNoneNanIndexLoc[xInd] = closestLowerNoneNanIndexLoc[xInd]+xSize;
		}
		if(closestUpperNoneNanIndexLoc[xInd]>xSize-1){ // avoid that we round out array bounds by using periodic boundaries
			closestUpperNoneNanIndexLoc[xInd] = closestUpperNoneNanIndexLoc[xInd]-xSize;
		}
	}while(NanValueLeft);
	
	//~ if(xInd>1998&xInd<2002){
		//~ printf("closestLowerNoneNanIndex[xInd], after: %d\n",closestLowerNoneNanIndexLoc[xInd]);
		//~ printf("closestUpperNoneNanIndex[xInd], after: %d\n",closestUpperNoneNanIndexLoc[xInd]);
	//~ }

	/* *****************************************************************
	 * interpolate locations that are NaN 
	 * ****************************************************************/
	//double distToLowerIndex = 0;
	//if(closestLowerNoneNanIndexLoc[xInd]>=0){
		//distToLowerIndex = convert_float( xInd - closestLowerNoneNanIndexLoc[xInd] );
	//}
	//else{ // wrap around
		//distToLowerIndex = convert_float( xInd + (xSize - closestLowerNoneNanIndexLoc[xInd]) );
	//}
	
	//double distToUpperIndex = 0;
	//if(closestUpperNoneNanIndexLoc[xInd]<xSize){
		//distToUpperIndex = convert_float( closestUpperNoneNanIndexLoc[xInd] - xInd );
	//}
	//else{ // wrap around
		//distToUpperIndex = convert_float( xInd + (closestUpperNoneNanIndexLoc[xInd] - xSize) );
	//}
	
	if(distToLowerIndex!=0 & distToUpperIndex!=0){
		//~ membraneCoordinatesX[xInd] = (distToLowerIndex * membraneCoordinatesX[closestLowerNoneNanIndexLoc[xInd]] + membraneCoordinatesX[closestUpperNoneNanIndexLoc[xInd]])/2;
		membraneCoordinatesX[xInd] = ((double)distToLowerIndex * membraneCoordinatesX[closestLowerNoneNanIndexLoc[xInd]] 
									+ (double)distToUpperIndex * membraneCoordinatesX[closestUpperNoneNanIndexLoc[xInd]])
									/(double)(distToLowerIndex+distToUpperIndex);
		//~ membraneCoordinatesY[xInd] = (membraneCoordinatesY[closestLowerNoneNanIndexLoc[xInd]] + membraneCoordinatesY[closestUpperNoneNanIndexLoc[xInd]])/2;
		membraneCoordinatesY[xInd] = (distToLowerIndex * membraneCoordinatesY[closestLowerNoneNanIndexLoc[xInd]] 
									+ distToUpperIndex * membraneCoordinatesY[closestUpperNoneNanIndexLoc[xInd]])
									/(distToLowerIndex+distToUpperIndex);
		
		//~ membraneNormalVectorsX[xInd] = (membraneNormalVectorsX[closestLowerNoneNanIndexLoc[xInd]] + membraneNormalVectorsX[closestUpperNoneNanIndexLoc[xInd]])/2;
		membraneNormalVectorsX[xInd] = ((double)distToLowerIndex * membraneNormalVectorsX[closestLowerNoneNanIndexLoc[xInd]] 
									  + (double)distToUpperIndex * membraneNormalVectorsX[closestUpperNoneNanIndexLoc[xInd]])
									  /(double)(distToLowerIndex+distToUpperIndex);

		//~ membraneNormalVectorsY[xInd] = (membraneNormalVectorsY[closestLowerNoneNanIndexLoc[xInd]] + membraneNormalVectorsY[closestUpperNoneNanIndexLoc[xInd]])/2;
		membraneNormalVectorsY[xInd] = ((double)distToLowerIndex * membraneNormalVectorsY[closestLowerNoneNanIndexLoc[xInd]] 
									  + (double)distToUpperIndex * membraneNormalVectorsY[closestUpperNoneNanIndexLoc[xInd]])
									  /(double)(distToLowerIndex+distToUpperIndex);

		//~ membraneNormalNorm = sqrt( pow(xMembraneNormalTmp,2) + pow(yMembraneNormalTmp,2) );
		
		double membraneNormalNorm;
		membraneNormalNorm = sqrt( pow(membraneNormalVectorsX[xInd],2) + pow(membraneNormalVectorsY[xInd],2) );
		//~ membraneNormalNorm = sqrt( pow(membraneNormalVectorsXpriv,2) + pow(membraneNormalVectorsYpriv,2) );
		
		membraneNormalVectorsX[xInd] = membraneNormalVectorsX[xInd]/membraneNormalNorm;
		membraneNormalVectorsY[xInd] = membraneNormalVectorsY[xInd]/membraneNormalNorm;
		//~ membraneNormalVectorsX[xInd] = membraneNormalVectorsXpriv/membraneNormalNorm;
		//~ membraneNormalVectorsY[xInd] = membraneNormalVectorsYpriv/membraneNormalNorm;
		
		//~ dbgOut[xInd] = distToLowerIndex;
		//~ dbgOut2[xInd] = distToUpperIndex;
	}

}

__kernel void checkIfCenterConverged(
									__global double2* contourCenter,
									__global double2* previousContourCenter,
									__global int* trackingFinished,
									const double centerTolerance
									) // will set self.dev_iterationFinished to true, when called
{
	// this function will set trackingFinished to 0 (FALSE), if the euclidian distance between any coordinate of the interpolated contours is >coordinateTolerance
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	//~ __private double distance;
	__private double xDistance, yDistance;
	if(xInd==0){
		//~ distance =  sqrt(  pow((contourCenter[xInd].x - previousContourCenter[xInd].x),2)
						 //~ + pow((contourCenter[xInd].y - previousContourCenter[xInd].y),2)
						 //~ );
		xDistance = fabs(contourCenter[xInd].x - previousContourCenter[xInd].x);
		yDistance = fabs(contourCenter[xInd].y - previousContourCenter[xInd].y);
		
		//~ if(xInd==100){
			//~ printf("coordinateTolerance: %f\n",coordinateTolerance);
			//~ printf("distance: %f\n",distance);
		//~ }
		
		//~ if(distance>centerTolerance){
			//~ trackingFinished[0] = 0;
		//~ }
		if((xDistance>centerTolerance)||(yDistance>centerTolerance)){
			trackingFinished[0] = 0;
		}
	}
	//~ printf("iterationFinished before setting to true: %d\n",iterationFinished[0]);
	//~ printf("iterationFinished: %d\n",iterationFinished[0]);
}

__kernel void calculateInterCoordinateAngles(__global double* interCoordinateAngles,
											 __global double* membraneCoordinatesX,
											 __global double* membraneCoordinatesY
											 //~ __global double* dbgOut,
											 //~ __global double* dbgOut2
											 )
{
	//~ A · B = A B cos θ = |A||B| cos θ

	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	//~ __private double2 basePoint = {membraneCoordinatesX[coordinateIndex],membraneCoordinatesY[coordinateIndex]};
	__private double2 dsVector1, dsVector2;
	
	if(xInd>0 && xInd<xSize-1){ // calculate interior gradients
		dsVector1.x = membraneCoordinatesX[xInd] - membraneCoordinatesX[xInd-1];
		dsVector1.y = membraneCoordinatesY[xInd] - membraneCoordinatesY[xInd-1];
		
		dsVector2.x = membraneCoordinatesX[xInd+1] - membraneCoordinatesX[xInd];
		dsVector2.y = membraneCoordinatesY[xInd+1] - membraneCoordinatesY[xInd];
	}
	else if(xInd==0){ // calculate edge gradient
		dsVector1.x = membraneCoordinatesX[xInd] - membraneCoordinatesX[xSize-1];
		dsVector1.y = membraneCoordinatesY[xInd] - membraneCoordinatesY[xSize-1];
		
		dsVector2.x = membraneCoordinatesX[xInd+1] - membraneCoordinatesX[xInd];
		dsVector2.y = membraneCoordinatesY[xInd+1] - membraneCoordinatesY[xInd];
	}
	else if(xInd==xSize-1){ // calculate edge gradient
		dsVector1.x = membraneCoordinatesX[xInd] - membraneCoordinatesX[xInd-1];
		dsVector1.y = membraneCoordinatesY[xInd] - membraneCoordinatesY[xInd-1];
		
		dsVector2.x = membraneCoordinatesX[0] - membraneCoordinatesX[xInd];
		dsVector2.y = membraneCoordinatesY[0] - membraneCoordinatesY[xInd];
	}

	dsVector1 = normalize(dsVector1);
	dsVector2 = normalize(dsVector2);
	//~ double dotProduct = dsVector1.x*dsVector2.x+dsVector1.y*dsVector2.y;
	double dotProduct = dot(dsVector1,dsVector2);
	interCoordinateAngles[xInd] = acos(dotProduct);
	
	if(dotProduct>=1){
		interCoordinateAngles[xInd] = 0;
	}
	if(dotProduct<=-1){
		interCoordinateAngles[xInd] = M_PI;
	}
}

//~ int spline(const int n, int end1, int end2, double slope1,__global double c[], __global double d[]){
	void findClosestGoodCoordinates(const int xInd,
									const int xSize,
									__local int* closestLowerCorrectIndexLoc,
									__local int* closestUpperCorrectIndexLoc,
									__local int* listOfGoodCoordinates,
									int* distToLowerIndex,
									int* distToUpperIndex
									)
	{
	//~ __private double distToLowerIndex = 0;
	//~ __private double distToUpperIndex = 0;
	__private bool incorrectValueLeft;

	do{
		incorrectValueLeft = false;
		//~ if(isnan(membraneCoordinatesX[closestLowerCorrectIndexLoc[xInd]])){
		if(listOfGoodCoordinates[closestLowerCorrectIndexLoc[xInd]] == 0){
			closestLowerCorrectIndexLoc[xInd] -= 1;
			(*distToLowerIndex)++;
			incorrectValueLeft = true;
		}
		
		//~ if(isnan(membraneCoordinatesX[closestUpperCorrectIndexLoc[xInd]])){
		if(listOfGoodCoordinates[closestUpperCorrectIndexLoc[xInd]] == 0){
			closestUpperCorrectIndexLoc[xInd] += 1;
			(*distToUpperIndex)++;
			incorrectValueLeft = true;
		}
		//~ id = getGlobalId();
		//~ output[id] = input[id] * input[id];
		if(closestLowerCorrectIndexLoc[xInd]<0){ // avoid that we round out array bounds by using periodic boundaries
			closestLowerCorrectIndexLoc[xInd] = closestLowerCorrectIndexLoc[xInd]+xSize;
		}
		if(closestUpperCorrectIndexLoc[xInd]>xSize-1){ // avoid that we round out array bounds by using periodic boundaries
			closestUpperCorrectIndexLoc[xInd] = closestUpperCorrectIndexLoc[xInd]-xSize;
		}
	}while(incorrectValueLeft);
}

void interpolateIncorrectCoordinates(const int xInd,
									const int xSize,
									__global double2 previousContourCenter[],
									__global double membraneCoordinatesX[],
									__global double membraneCoordinatesY[],
									__global double membraneNormalVectorsX[],
									__global double membraneNormalVectorsY[],
									__local int* closestLowerCorrectIndexLoc,
									__local int* closestUpperCorrectIndexLoc,
									int* distToLowerIndex,
									int* distToUpperIndex
									)
{
	if(*distToLowerIndex!=0 & *distToUpperIndex!=0){
		membraneCoordinatesX[xInd] = ((double)*distToLowerIndex * membraneCoordinatesX[closestLowerCorrectIndexLoc[xInd]] 
									+ (double)*distToUpperIndex * membraneCoordinatesX[closestUpperCorrectIndexLoc[xInd]])
									/(double)(*distToLowerIndex+*distToUpperIndex);
		membraneCoordinatesY[xInd] = ((double)*distToLowerIndex * membraneCoordinatesY[closestLowerCorrectIndexLoc[xInd]] 
									+ (double)*distToUpperIndex * membraneCoordinatesY[closestUpperCorrectIndexLoc[xInd]])
									/(double)(*distToLowerIndex+*distToUpperIndex);
		
		//~ membraneNormalVectorsX[xInd] = (*distToLowerIndex * membraneNormalVectorsX[closestLowerCorrectIndexLoc[xInd]] 
									  //~ + *distToUpperIndex * membraneNormalVectorsX[closestUpperCorrectIndexLoc[xInd]])
									  //~ /(*distToLowerIndex+*distToUpperIndex);
//~ 
		//~ membraneNormalVectorsY[xInd] = (*distToLowerIndex * membraneNormalVectorsY[closestLowerCorrectIndexLoc[xInd]] 
									  //~ + *distToUpperIndex * membraneNormalVectorsY[closestUpperCorrectIndexLoc[xInd]])
									  //~ /(*distToLowerIndex+*distToUpperIndex);

		membraneNormalVectorsX[xInd] = membraneCoordinatesX[xInd] - previousContourCenter[0].x;
		membraneNormalVectorsY[xInd] = membraneCoordinatesY[xInd] - previousContourCenter[0].y;

		double membraneNormalNorm;
		membraneNormalNorm = sqrt( pow(membraneNormalVectorsX[xInd],2) + pow(membraneNormalVectorsY[xInd],2) );
		
		membraneNormalVectorsX[xInd] = membraneNormalVectorsX[xInd]/membraneNormalNorm;
		membraneNormalVectorsY[xInd] = membraneNormalVectorsY[xInd]/membraneNormalNorm;
		
		//~ dbgOut[xInd] = distToLowerIndex;
		//~ dbgOut2[xInd] = distToUpperIndex;
	}	
	
}

//~ self.listOfGoodCoordinates_memSize
__kernel void filterJumpedCoordinates(
											__global double2* previousContourCenter,
											__global double* membraneCoordinatesX,
											__global double* membraneCoordinatesY,
											__global double* membraneNormalVectorsX,
											__global double* membraneNormalVectorsY,
											__global double* previousInterpolatedMembraneCoordinatesX,
											__global double* previousInterpolatedMembraneCoordinatesY,
											__local int* closestLowerCorrectIndexLoc,
											__local int* closestUpperCorrectIndexLoc,
											__local int* listOfGoodCoordinates,
											const double maxCoordinateShift,
											__global int* dbg_listOfGoodCoordinates
											) // will set self.dev_iterationFinished to true, when called
{
	// this function will set trackingFinished to 0 (FALSE), if the euclidian distance between any coordinate of the interpolated contours is >coordinateTolerance
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	listOfGoodCoordinates[xInd] = 1;
	closestLowerCorrectIndexLoc[xInd] = xInd;
	closestUpperCorrectIndexLoc[xInd] = xInd;
	__private double distance;
	
	distance =  sqrt(  pow((membraneCoordinatesX[xInd] - previousInterpolatedMembraneCoordinatesX[xInd]),2)
					 + pow((membraneCoordinatesY[xInd] - previousInterpolatedMembraneCoordinatesY[xInd]),2)
					 );
	
	//~ if(xInd==100){
		//~ printf("coordinateTolerance: %f\n",coordinateTolerance);
		//~ printf("distance: %f\n",distance);
	//~ }
	
	//~ if(distance>coordinateTolerance){
		//~ trackingFinished[0] = 0;
		//~ printf("xInd: %d\n",xInd);
		//~ printf("distance: %f\n",distance);
	//~ }
	
	//~ printf("iterationFinished before setting to true: %d\n",iterationFinished[0]);
	//~ printf("iterationFinished: %d\n",iterationFinished[0]);
	
	if(distance>maxCoordinateShift){
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
							  
	dbg_listOfGoodCoordinates[xInd] = listOfGoodCoordinates[xInd];
	
	interpolateIncorrectCoordinates(xInd,
									xSize,
									previousContourCenter,
									membraneCoordinatesX,
									membraneCoordinatesY,
									membraneNormalVectorsX,
									membraneNormalVectorsY,
									closestLowerCorrectIndexLoc,
									closestUpperCorrectIndexLoc,
									&distToLowerIndex,
									&distToUpperIndex
									);
}

__kernel void filterIncorrectCoordinates(__global double2* previousContourCenter,
										 __global double* interCoordinateAngles,
										 __global double* membraneCoordinatesX,
										 __global double* membraneCoordinatesY,
										 __global double* membraneNormalVectorsX,
										 __global double* membraneNormalVectorsY,
										 __local int* closestLowerCorrectIndexLoc,
										 __local int* closestUpperCorrectIndexLoc,
										 const double maxInterCoordinateAngle
										 //~ __global double* dbgOut,
										 //~ __global double* dbgOut2
										 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	__private bool incorrectValueLeft;
	//~ __local int closestLowerCorrectIndexLoc[2048];
	//~ __local int closestUpperCorrectIndexLoc[2048];
	
	//~ closestLowerCorrectIndexLoc[xInd] = closestLowerCorrectIndex[xInd];
	//~ closestUpperCorrectIndexLoc[xInd] = closestUpperCorrectIndex[xInd];
	
	closestLowerCorrectIndexLoc[xInd] = xInd;
	closestUpperCorrectIndexLoc[xInd] = xInd;
	
	//~ if(xInd>1998&xInd<2002){
		//~ printf("closestLowerCorrectIndex[xInd], before: %d\n",closestLowerCorrectIndexLoc[xInd]);
		//~ printf("closestUpperCorrectIndex[xInd], before: %d\n",closestUpperCorrectIndexLoc[xInd]);
	//~ }
	
	__private int distToLowerIndex = 0;
	__private int distToUpperIndex = 0;
	do{
		incorrectValueLeft = false;
		//~ if(isnan(membraneCoordinatesX[closestLowerCorrectIndexLoc[xInd]])){
		if(interCoordinateAngles[closestLowerCorrectIndexLoc[xInd]] > maxInterCoordinateAngle){
			closestLowerCorrectIndexLoc[xInd] -= 1;
			distToLowerIndex++;
			incorrectValueLeft = true;
		}
		
		//~ if(isnan(membraneCoordinatesX[closestUpperCorrectIndexLoc[xInd]])){
		if(interCoordinateAngles[closestUpperCorrectIndexLoc[xInd]] > maxInterCoordinateAngle){
			closestUpperCorrectIndexLoc[xInd] += 1;
			distToUpperIndex++;
			incorrectValueLeft = true;
		}
		//~ id = getGlobalId();
		//~ output[id] = input[id] * input[id];
		if(closestLowerCorrectIndexLoc[xInd]<0){ // avoid that we round out array bounds by using periodic boundaries
			closestLowerCorrectIndexLoc[xInd] = closestLowerCorrectIndexLoc[xInd]+xSize;
		}
		if(closestUpperCorrectIndexLoc[xInd]>xSize-1){ // avoid that we round out array bounds by using periodic boundaries
			closestUpperCorrectIndexLoc[xInd] = closestUpperCorrectIndexLoc[xInd]-xSize;
		}
	}while(incorrectValueLeft);
	
	/* *****************************************************************
	 * interpolate locations that are NaN 
	 * ****************************************************************/
	
	if(distToLowerIndex!=0 & distToUpperIndex!=0){
		membraneCoordinatesX[xInd] = ((double)distToLowerIndex * membraneCoordinatesX[closestLowerCorrectIndexLoc[xInd]] 
									+ (double)distToUpperIndex * membraneCoordinatesX[closestUpperCorrectIndexLoc[xInd]])
									/(double)(distToLowerIndex+distToUpperIndex);
		membraneCoordinatesY[xInd] = ((double)distToLowerIndex * membraneCoordinatesY[closestLowerCorrectIndexLoc[xInd]] 
									+ (double)distToUpperIndex * membraneCoordinatesY[closestUpperCorrectIndexLoc[xInd]])
									/(double)(distToLowerIndex+distToUpperIndex);
		
		//~ membraneNormalVectorsX[xInd] = (distToLowerIndex * membraneNormalVectorsX[closestLowerCorrectIndexLoc[xInd]] 
									  //~ + distToUpperIndex * membraneNormalVectorsX[closestUpperCorrectIndexLoc[xInd]])
									  //~ /(distToLowerIndex+distToUpperIndex);
		//~ membraneNormalVectorsY[xInd] = (distToLowerIndex * membraneNormalVectorsY[closestLowerCorrectIndexLoc[xInd]] 
									  //~ + distToUpperIndex * membraneNormalVectorsY[closestUpperCorrectIndexLoc[xInd]])
									  //~ /(distToLowerIndex+distToUpperIndex);

		membraneNormalVectorsX[xInd] = membraneCoordinatesX[xInd] - previousContourCenter[0].x;
		membraneNormalVectorsY[xInd] = membraneCoordinatesY[xInd] - previousContourCenter[0].y;
		
		double membraneNormalNorm;
		membraneNormalNorm = sqrt( pow(membraneNormalVectorsX[xInd],2) + pow(membraneNormalVectorsY[xInd],2) );
		//~ membraneNormalNorm = sqrt( pow(membraneNormalVectorsXpriv,2) + pow(membraneNormalVectorsYpriv,2) );
		
		membraneNormalVectorsX[xInd] = membraneNormalVectorsX[xInd]/membraneNormalNorm;
		membraneNormalVectorsY[xInd] = membraneNormalVectorsY[xInd]/membraneNormalNorm;
		//~ membraneNormalVectorsX[xInd] = membraneNormalVectorsXpriv/membraneNormalNorm;
		//~ membraneNormalVectorsY[xInd] = membraneNormalVectorsYpriv/membraneNormalNorm;
		
		//~ dbgOut[xInd] = distToLowerIndex;
		//~ dbgOut2[xInd] = distToUpperIndex;
	}

}

__kernel void interpolatePolarCoordinatesLinear(__global double* membranePolarRadius,
											  __global double* membranePolarTheta,
											  __global double2* radialVectors,
											  __global double2* contourCenter,
											  __global double* membraneCoordinatesX,
											  __global double* membraneCoordinatesY,
											  __global double* interpolatedMembraneCoordinatesX,
											  __global double* interpolatedMembraneCoordinatesY,
											  //~ __local double* membranePolarRadiusLoc,
											  //~ __local double* membranePolarThetaLoc,
											  __global double* membranePolarRadiusLoc,
											  __global double* membranePolarThetaLoc,
											  //~ __local double* interpolationMembranePolarRadius,
											  //~ __local double* interpolationMembranePolarTheta,
											  __global double* interpolationMembranePolarRadius,
											  __global double* interpolationMembranePolarTheta,
											  __global double* interpolationMembranePolarRadiusTesting,
											  __global double* interpolationMembranePolarThetaTesting,
											  __global double* interpolationAngles,
											  const int nrOfInterpolationPoints,
											  const int nrOfContourPoints,
											  const int nrOfAnglesToCompare,
											  __global double* dbgOut,
											  __global double* dbgOut2
											 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	dbgOut[xInd] = 0;
	dbgOut2[xInd] = 0;
	
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
		
		lowerAngle = membranePolarTheta[lowerIndex];
		upperAngle = membranePolarTheta[upperIndex];
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
			radialVector.x = membraneCoordinatesX[lowerIndex] - radialLineBasePoint.x;
			radialVector.y = membraneCoordinatesY[lowerIndex] - radialLineBasePoint.y;
			if(length(radialVector)<distanceFromCenter){
				distanceFromCenter = length(radialVector);
				interpolatedMembranePoint.x = membraneCoordinatesX[lowerIndex];
				interpolatedMembranePoint.y = membraneCoordinatesY[lowerIndex];
			}
			radialVector.x = membraneCoordinatesX[upperIndex] - radialLineBasePoint.x;
			radialVector.y = membraneCoordinatesY[upperIndex] - radialLineBasePoint.y;
			if(length(radialVector)<distanceFromCenter){
				distanceFromCenter = length(radialVector);
				interpolatedMembranePoint.x = membraneCoordinatesX[upperIndex];
				interpolatedMembranePoint.y = membraneCoordinatesY[upperIndex];
			}
			break;
		}
		
		//~ if( membranePolarTheta[xInd] >= interpolationAngles[lowerIndex] && membranePolarTheta[xInd] < interpolationAngles[upperIndex] )
		if( interpolationAngle >= lowerAngle && interpolationAngle < upperAngle )
		{
			dbgOut[xInd] = lowerAngle;
			dbgOut2[xInd] = upperAngle;

			// lineSegmentDirectionVector <-> e
			// lineSegmentBasePoint <-> f
			__private double2 lineSegmentDirectionVector;
			__private double2 lineSegmentBasePoint;
			
			// calculate paramaters of the straight, that passes through contour-segment
			lineSegmentDirectionVector.x = membraneCoordinatesX[upperIndex] - membraneCoordinatesX[lowerIndex];
			lineSegmentDirectionVector.y = membraneCoordinatesY[upperIndex] - membraneCoordinatesY[lowerIndex];
			lineSegmentDirectionVector = normalize(lineSegmentDirectionVector);
						
			lineSegmentBasePoint.x = membraneCoordinatesX[lowerIndex];
			lineSegmentBasePoint.y = membraneCoordinatesY[lowerIndex];
			
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
			
			radialVector.x = interpolatedMembranePointTMP.x - radialLineBasePoint.x;
			radialVector.y = interpolatedMembranePointTMP.y - radialLineBasePoint.y;
			if(length(radialVector)<distanceFromCenter){
				distanceFromCenter = length(radialVector);
				//~ interpolatedMembranePoint = {membraneCoordinatesX[lowerIndex],membraneCoordinatesY[lowerIndex]};
				interpolatedMembranePoint = interpolatedMembranePointTMP;
			}
			break;
		}

	}
	interpolatedMembraneCoordinatesX[xInd] = interpolatedMembranePoint.x;
	interpolatedMembraneCoordinatesY[xInd] = interpolatedMembranePoint.y;
	//barrier(CLK_LOCAL_MEM_FENCE);
	//barrier(CLK_GLOBAL_MEM_FENCE);

}
