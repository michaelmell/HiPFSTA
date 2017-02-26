#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif
//~ #pragma OPENCL EXTENSION cl_amd_printf : enable

void fit(__constant double x[], double y[], int gradientCenterIndex, int linFitParameter, double *a, double *b, double *siga, double *sigb, double *chi2)
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

__kernel void findMembranePosition(sampler_t sampler, 
								   __read_only image2d_t Img,
								   const int imgSizeX,
								   const int imgSizeY,
								   __constant double* localRotationMatrices, // this should be local or constant
								   __constant double* linFitSearchRangeXvalues, // this should be local or constant
								   const int linFitParameter,
								   //~ __local double* fitIntercept,
								   //~ __local double* fitIncline,
								   __global double* fitIntercept,
								   __global double* fitIncline,
								   const int meanParameter,
								   __constant double* meanRangeXvalues, // this should be local or constant
								   const double meanRangePositionOffset,
								   //~ __local double* localMembranePositionsX,
								   //~ __local double* localMembranePositionsY,
								   __global double* localMembranePositionsX,
								   __global double* localMembranePositionsY,
								   __global double* membraneCoordinatesX,
								   __global double* membraneCoordinatesY,
								   __global double* membraneNormalVectorsX,
								   __global double* membraneNormalVectorsY,
								   const int coordinateIndex)
{
	const int xInd = get_global_id(1);
	const int yInd = get_global_id(0);
	const int xSize = get_global_size(1);
	const int ySize = get_global_size(0);
	
	//~ const int xIndLoc = get_local_id(1);
	//~ const int yIndLoc = get_local_id(0);
	
	__private double lineIntensities[400];
	
	__private double2 membraneNormalVector;
	membraneNormalVector.x = membraneNormalVectorsX[coordinateIndex];
	membraneNormalVector.y = membraneNormalVectorsY[coordinateIndex];
	
	// matrix multiplication with linear array of sequential 2x2 rotation matrices
	__private double2 rotatedUnitVector;
	rotatedUnitVector.x = localRotationMatrices[4*xInd+0] * membraneNormalVector.x
	                     + localRotationMatrices[4*xInd+1] * membraneNormalVector.y;
	rotatedUnitVector.y = localRotationMatrices[4*xInd+2] * membraneNormalVector.x
	                     + localRotationMatrices[4*xInd+3] * membraneNormalVector.y;
	
	barrier(CLK_GLOBAL_MEM_FENCE);

	__private int maxIndex;
	__private int minIndex;
	__private double minValue = 32000; // initialize value with first value of array
	__private double maxValue = 0; // initialize value with first value of array
	
	__private double2 Coords;
	__private double2 NormCoords;
	__private const int2 dims = get_image_dim(Img);
	
	//~ if(xInd==0 && yInd==0){
		//~ printf("dims.x: %d\n",dims.x);
		//~ printf("dims.y: %d\n",dims.y);
	//~ }
	
	__private double2 basePoint = {membraneCoordinatesX[coordinateIndex],membraneCoordinatesY[coordinateIndex]};

	barrier(CLK_GLOBAL_MEM_FENCE);

	for(int index=0;index<imgSizeY;index++){
		Coords.x = basePoint.x + rotatedUnitVector.x * linFitSearchRangeXvalues[index];
		Coords.y = basePoint.y + rotatedUnitVector.y * linFitSearchRangeXvalues[index];
		
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

	//~ __private int gradientCenterIndex, tmp;
	__private int gradientCenterIndex;
	__private double gradientCenterValue = minValue+(maxValue-minValue)/2.0;
	__private double minValue2 = 20000;
	__private double refValue;
	
	__private double a=0.0, b=0.0, siga=0.0, sigb=0.0, chi2=0.0;
		
	barrier(CLK_GLOBAL_MEM_FENCE);

	if(minIndex<maxIndex){
		for(int index=minIndex;index<maxIndex;index++){
			refValue = fabs(lineIntensities[index]-gradientCenterValue);
			gradientCenterIndex = select(gradientCenterIndex,index,(minValue2 > refValue));
			minValue2 = select(minValue2,refValue,(long)(minValue2 > refValue));
		}
		// "r = a > b ? a : b" corresponds to: "r = select(b, a, a > b)", corresponds to "if(a>b){r=a}; else{r=b}"
		// reference: http://stackoverflow.com/questions/7635706/opencl-built-in-function-select
		
		fit(linFitSearchRangeXvalues, lineIntensities, gradientCenterIndex, linFitParameter, &a, &b, &siga, &sigb, &chi2);
		fitIntercept[xInd] = a;
		fitIncline[xInd] = b;
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);

	__private double meanIntensity = 0.0;
	for(int index=0;index<meanParameter;index++){
		Coords.x = basePoint.x + rotatedUnitVector.x * ( meanRangeXvalues[index] + meanRangePositionOffset );
		Coords.y = basePoint.y + rotatedUnitVector.y * ( meanRangeXvalues[index] + meanRangePositionOffset );
		
		NormCoords = Coords/convert_double2(dims);
		
		float2 fNormCoords;
		fNormCoords.x = (float)NormCoords.x;
		fNormCoords.y = (float)NormCoords.y;

		meanIntensity = meanIntensity + read_imagef(Img, sampler, fNormCoords).x;
	}
	meanIntensity = meanIntensity/convert_float(meanParameter);
	
	barrier(CLK_GLOBAL_MEM_FENCE);

	__private double relativeMembranePositionLocalCoordSys;
	//~ __private double membranePositionsX, membranePositionsY;
	relativeMembranePositionLocalCoordSys = (meanIntensity-fitIntercept[xInd])/fitIncline[xInd];
	
	//~ if(xInd==0 && yInd==0){
		//~ printf("relativeMembranePositionLocalCoordSys: %f\n",relativeMembranePositionLocalCoordSys);
		//~ printf("meanIntensity: %f\n",meanIntensity);
	//~ }
	
	localMembranePositionsX[xInd] = basePoint.x + rotatedUnitVector.x * relativeMembranePositionLocalCoordSys;
	localMembranePositionsY[xInd] = basePoint.y + rotatedUnitVector.y * relativeMembranePositionLocalCoordSys;
	
	//~ if(xInd==0 && coordinateIndex==200){
		//~ printf("localMembranePositionsX[xInd]: %f\n",localMembranePositionsX[xInd]);
		//~ printf("localMembranePositionsY[xInd]: %f\n",localMembranePositionsY[xInd]);
	//~ }
	
	//~ barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	__private double xTmp = 0.0, yTmp = 0.0, meanIncline = 0.0;
	if(xInd==0){
		//~ for(int index=get_local_id(1);index<xSize;index++){
		for(int index=0;index<xSize;index++){
			//~ xTmp += localMembranePositionsX[index];
			//~ yTmp += localMembranePositionsY[index];
			xTmp += fitIncline[index] * localMembranePositionsX[index];
			yTmp += fitIncline[index] * localMembranePositionsY[index];
			meanIncline += fitIncline[index];
		}
		//~ membraneCoordinatesX[coordinateIndex] = xTmp/convert_float(imgSizeX);
		//~ membraneCoordinatesY[coordinateIndex] = yTmp/convert_float(imgSizeX);
		membraneCoordinatesX[coordinateIndex] = xTmp/meanIncline;
		membraneCoordinatesY[coordinateIndex] = yTmp/meanIncline;
	}
}

__kernel void calculateMembraneNormalVectors(__global double* membraneCoordinatesX,
											 __global double* membraneCoordinatesY,
											 __global double* membraneNormalVectorsX,
											 __global double* membraneNormalVectorsY
											 //~ __local double2* membraneNormalVectors
										)
	{
		const int xInd = get_global_id(0);
		const int xSize = get_global_size(0);
		
		__private double vectorNorm;
		
		// NOTE: we use bilinear interpolation to calculate the gradient vectors
		if(xInd>0 && xInd<xSize-1){ // calculate interior gradients
			//~ membraneNormalVectors[xInd].y = -(  (membraneCoordinatesX[xInd] - membraneCoordinatesX[xInd-1])
			                                  //~ + (membraneCoordinatesX[xInd+1] - membraneCoordinatesX[xInd]) )/2;
			//~ membraneNormalVectors[xInd].x =  (  (membraneCoordinatesY[xInd] - membraneCoordinatesY[xInd-1])
			                                  //~ + (membraneCoordinatesY[xInd+1] - membraneCoordinatesY[xInd]) )/2;
			membraneNormalVectorsY[xInd] = -(  (membraneCoordinatesX[xInd] - membraneCoordinatesX[xInd-1])
			                                  + (membraneCoordinatesX[xInd+1] - membraneCoordinatesX[xInd]) )/2;
			membraneNormalVectorsX[xInd] =  (  (membraneCoordinatesY[xInd] - membraneCoordinatesY[xInd-1])
			                                  + (membraneCoordinatesY[xInd+1] - membraneCoordinatesY[xInd]) )/2;
		}
		else if(xInd==0){ // calculate edge gradient
			//~ membraneNormalVectors[xInd].y = -(  (membraneCoordinatesX[xInd] - membraneCoordinatesX[xSize-1])
			                                  //~ + (membraneCoordinatesX[xInd+1] - membraneCoordinatesX[xInd]) )/2;
			//~ membraneNormalVectors[xInd].x =  (  (membraneCoordinatesY[xInd] - membraneCoordinatesY[xSize-1])
			                                  //~ + (membraneCoordinatesY[xInd+1] - membraneCoordinatesY[xInd]) )/2;
			membraneNormalVectorsY[xInd] = -(  (membraneCoordinatesX[xInd] - membraneCoordinatesX[xSize-1])
			                                  + (membraneCoordinatesX[xInd+1] - membraneCoordinatesX[xInd]) )/2;
			membraneNormalVectorsX[xInd] =  (  (membraneCoordinatesY[xInd] - membraneCoordinatesY[xSize-1])
			                                  + (membraneCoordinatesY[xInd+1] - membraneCoordinatesY[xInd]) )/2;
		}
		else if(xInd==xSize-1){ // calculate edge gradient
			//~ membraneNormalVectors[xInd].y = -(  (membraneCoordinatesX[xInd] - membraneCoordinatesX[xInd-1])
			                                  //~ + (membraneCoordinatesX[0] - membraneCoordinatesX[xInd]) )/2;
			//~ membraneNormalVectors[xInd].x =  (  (membraneCoordinatesY[xInd] - membraneCoordinatesY[xInd-1])
			                                  //~ + (membraneCoordinatesY[0] - membraneCoordinatesY[xInd]) )/2;
			membraneNormalVectorsY[xInd] = -(  (membraneCoordinatesX[xInd] - membraneCoordinatesX[xInd-1])
			                                  + (membraneCoordinatesX[0] - membraneCoordinatesX[xInd]) )/2;
			membraneNormalVectorsX[xInd] =  (  (membraneCoordinatesY[xInd] - membraneCoordinatesY[xInd-1])
			                                  + (membraneCoordinatesY[0] - membraneCoordinatesY[xInd]) )/2;
		}
		
		barrier(CLK_GLOBAL_MEM_FENCE);
		//~ vectorNorm = sqrt(pow(membraneNormalVectors[xInd].x,2) + pow(membraneNormalVectors[xInd].y,2));
		vectorNorm = sqrt(pow(membraneNormalVectorsX[xInd],2) + pow(membraneNormalVectorsY[xInd],2));
		
		//~ membraneNormalVectorsX[xInd] = membraneNormalVectors[xInd].x/vectorNorm;
		//~ membraneNormalVectorsY[xInd] = membraneNormalVectors[xInd].y/vectorNorm;
		membraneNormalVectorsX[xInd] = membraneNormalVectorsX[xInd]/vectorNorm;
		membraneNormalVectorsY[xInd] = membraneNormalVectorsY[xInd]/vectorNorm;
	}

__kernel void calculateContourCenter(__global double* membraneCoordinatesX,
									 __global double* membraneCoordinatesY,
									 __local double* ds,
									 __local double* sumds,
									 __global double2* contourCenter
									 //const double2 contourCenter
									 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	const int xIndLoc = get_local_id(0);
	const int xSizeLoc = get_local_size(0);
	__private double circumference=0.0;
	
	
	if(xInd>=1){
		ds[xIndLoc] = sqrt(pow((membraneCoordinatesX[xIndLoc] - membraneCoordinatesX[xIndLoc-1]),2)
				  + 	   pow((membraneCoordinatesY[xIndLoc] - membraneCoordinatesY[xIndLoc-1]),2)
				  );
	}
	else if(xInd==0){ // calculate edge gradient
		ds[xIndLoc] = sqrt(pow((membraneCoordinatesX[xIndLoc] - membraneCoordinatesX[xSizeLoc-1]),2)
				  + 	   pow((membraneCoordinatesY[xIndLoc] - membraneCoordinatesY[xSizeLoc-1]),2)
				  );
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	if(xInd>=1){
		sumds[xIndLoc] = ds[xIndLoc] + ds[xIndLoc-1];
	}
	else if(xInd==0){ // calculate edge gradient
		sumds[xIndLoc] = ds[xIndLoc] + ds[xSizeLoc-1];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	
	__private double tmp1=0.0, tmp2=0.0;
	if(xIndLoc==0){
		for(int index=0;index<xSize;index++){
			circumference = circumference + ds[index];
		}
		
		for(int index=0;index<xSize;index++){
			tmp1 = tmp1 + membraneCoordinatesX[index] * sumds[index];
			tmp2 = tmp2 + membraneCoordinatesY[index] * sumds[index];
		}
		contourCenter[xIndLoc].x = (1/(2*circumference)) * tmp1;
		contourCenter[xIndLoc].y = (1/(2*circumference)) * tmp2;
	}
}

__kernel void calculateContourCenterNew(__global double* membraneCoordinatesX,
									 __global double* membraneCoordinatesY,
									 //~ __local double* ds,
									 //~ __local double* sumds,
									 __global double* ds,
									 __global double* sumds,
									 __global double2* contourCenter
									 //~ const double2 contourCenter
									 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	//~ const int xIndLoc = get_local_id(0);
	//~ const int xSizeLoc = get_local_size(0);
	__private double circumference=0.0;
	
	
	if(xInd>=1){
		ds[xInd] = sqrt(pow((membraneCoordinatesX[xInd] - membraneCoordinatesX[xInd-1]),2)
				  + 	pow((membraneCoordinatesY[xInd] - membraneCoordinatesY[xInd-1]),2)
				  );
	}
	else if(xInd==0){ // calculate edge gradient
		ds[xInd] = sqrt(pow((membraneCoordinatesX[xInd] - membraneCoordinatesX[xSize-1]),2)
				  + 	pow((membraneCoordinatesY[xInd] - membraneCoordinatesY[xSize-1]),2)
				  );
	}
	
	//~ barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);

	if(xInd>=1){
		sumds[xInd] = ds[xInd] + ds[xInd-1];
	}
	else if(xInd==0){ // calculate edge gradient
		sumds[xInd] = ds[xInd] + ds[xSize-1];
	}

	//~ barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);

	__private double tmp1=0.0, tmp2=0.0;
	if(xInd==0){
		for(int index=0;index<xSize;index++){
			circumference = circumference + ds[index];
		}
		
		for(int index=0;index<xSize;index++){
			tmp1 = tmp1 + membraneCoordinatesX[index] * sumds[index];
			tmp2 = tmp2 + membraneCoordinatesY[index] * sumds[index];
		}
		contourCenter[xInd].x = (1/(2*circumference)) * tmp1;
		contourCenter[xInd].y = (1/(2*circumference)) * tmp2;
		
		//~ printf("circumference: %f\n",circumference);
		//~ printf("tmp1: %f\n",tmp1);
		//~ printf("tmp2: %f\n",tmp2);
		//~ printf("contourCenter[xInd].x: %f\n",contourCenter[xInd].x);
		//~ printf("contourCenter[xInd].y: %f\n",contourCenter[xInd].y);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}


__kernel void calculateDs(__global double* membraneCoordinatesX,
						 __global double* membraneCoordinatesY,
						 __global double* ds
						 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);

	if(xInd>=1){
		ds[xInd] = sqrt(pow((membraneCoordinatesX[xInd] - membraneCoordinatesX[xInd-1]),2)
				  + 	pow((membraneCoordinatesY[xInd] - membraneCoordinatesY[xInd-1]),2)
				  );
	}
	else if(xInd==0){ // calculate edge gradient
		ds[xInd] = sqrt(pow((membraneCoordinatesX[xInd] - membraneCoordinatesX[xSize-1]),2)
				  + 	pow((membraneCoordinatesY[xInd] - membraneCoordinatesY[xSize-1]),2)
				  );
	}
}

__kernel void calculateSumDs(__global double* membraneCoordinatesX,
									 __global double* membraneCoordinatesY,
									 __global double* ds,
									 __global double* sumds
									 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	//~ const int xIndLoc = get_local_id(0);
	//~ const int xSizeLoc = get_local_size(0);
	//~ __private double circumference=0.0;
		
	//~ if(xInd>=1){
		//~ ds[xInd] = sqrt(pow((membraneCoordinatesX[xInd] - membraneCoordinatesX[xInd-1]),2)
				  //~ + 	pow((membraneCoordinatesY[xInd] - membraneCoordinatesY[xInd-1]),2)
				  //~ );
	//~ }
	//~ else if(xInd==0){ // calculate edge gradient
		//~ ds[xInd] = sqrt(pow((membraneCoordinatesX[xInd] - membraneCoordinatesX[xSize-1]),2)
				  //~ + 	pow((membraneCoordinatesY[xInd] - membraneCoordinatesY[xSize-1]),2)
				  //~ );
	//~ }
	
	//~ barrier(CLK_LOCAL_MEM_FENCE);
	//~ barrier(CLK_GLOBAL_MEM_FENCE);
	//~ write_mem_fence (CLK_GLOBAL_MEM_FENCE);
	
	if(xInd>=1){
		sumds[xInd] = ds[xInd] + ds[xInd-1];
	}
	else if(xInd==0){ // calculate edge gradient
		sumds[xInd] = ds[xInd] + ds[xSize-1];
	}

	//~ barrier(CLK_GLOBAL_MEM_FENCE);
	//~ write_mem_fence (CLK_GLOBAL_MEM_FENCE);

	//~ barrier(CLK_LOCAL_MEM_FENCE);
	//~ barrier(CLK_GLOBAL_MEM_FENCE);

	//__private double tmp1=0.0, tmp2=0.0;
	//if(xInd==0){
		//for(int index=0;index<xSize;index++){
			//circumference = circumference + ds[index];
		//}
		
		//for(int index=0;index<xSize;index++){
			//tmp1 = tmp1 + membraneCoordinatesX[index] * sumds[index];
			//tmp2 = tmp2 + membraneCoordinatesY[index] * sumds[index];
		//}
		//contourCenter[xInd].x = (1/(2*circumference)) * tmp1;
		//contourCenter[xInd].y = (1/(2*circumference)) * tmp2;
		
		////~ printf("circumference: %f\n",circumference);
		////~ printf("tmp1: %f\n",tmp1);
		////~ printf("tmp2: %f\n",tmp2);
		////~ printf("contourCenter[xInd].x: %f\n",contourCenter[xInd].x);
		////~ printf("contourCenter[xInd].y: %f\n",contourCenter[xInd].y);
	//}

	//~ barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void calculateContourCenterNew2(__global double* membraneCoordinatesX,
										 __global double* membraneCoordinatesY,
										 __global double* ds,
										 __global double* sumds,
										 __global double2* contourCenter,
										 const int nrOfContourPoints
										 //~ const double2 contourCenter
									 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	//~ const int xIndLoc = get_local_id(0);
	//~ const int xSizeLoc = get_local_size(0);
	__private double circumference=0.0;
	
	__private double tmp1=0.0, tmp2=0.0;
	if(xInd==0){
		for(int index=0;index<nrOfContourPoints;index++){
			//~ printf("index: %d\n",index);
			circumference = circumference + ds[index];
		}
		
		for(int index=0;index<nrOfContourPoints;index++){
			tmp1 = tmp1 + membraneCoordinatesX[index] * sumds[index];
			tmp2 = tmp2 + membraneCoordinatesY[index] * sumds[index];
		}
		contourCenter[0].x = (1/(2*circumference)) * tmp1;
		contourCenter[0].y = (1/(2*circumference)) * tmp2;
		
		//~ printf("circumference: %f\n",circumference);
		//~ printf("tmp1: %f\n",tmp1);
		//~ printf("tmp2: %f\n",tmp2);
		//~ printf("contourCenter[xInd].x: %f\n",contourCenter[xInd].x);
		//~ printf("contourCenter[xInd].y: %f\n",contourCenter[xInd].y);
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


//~ int spline (int n, int end1, int end2, double slope1, double slope2, double x[], double y[], double b[], double c, double d[], int *iflag)

//~ int spline (n, end1, end2, slope1, slope2, x, y, b, c, d, iflag)
//~ 
//~ int    n, end1, end2;
//~ double slope1, slope2;
//~ double x[], y[], b[], c[], d[];
//~ int    *iflag;

int spline(const int n, int end1, int end2, double slope1, double slope2, __global double x[], __global double y[], __global double b[], __global double c[], __global double d[], int* iflag)
{  /* begin procedure spline() */

int    nm1, ib, i;
double t;
int    ascend;

nm1    = n - 1;
*iflag = 0;

if (n < 2)
  {  /* no possible interpolation */
  *iflag = 1;
  goto LeaveSpline;
  }

ascend = 1;
for (i = 1; i < n; ++i) if (x[i] <= x[i-1]) ascend = 0;
if (!ascend)
   {
   *iflag = 2;
   goto LeaveSpline;
   }

if (n >= 3)
   {    /* ---- At least quadratic ---- */

   /* ---- Set up the symmetric tri-diagonal system
           b = diagonal
           d = offdiagonal
           c = right-hand-side  */
   d[0] = x[1] - x[0];
   c[1] = (y[1] - y[0]) / d[0];
   for (i = 1; i < nm1; ++i)
      {
      d[i]   = x[i+1] - x[i];
      b[i]   = 2.0 * (d[i-1] + d[i]);
      c[i+1] = (y[i+1] - y[i]) / d[i];
      c[i]   = c[i+1] - c[i];
      }

   /* ---- Default End conditions
           Third derivatives at x[0] and x[n-1] obtained
           from divided differences  */
   b[0]   = -d[0];
   b[nm1] = -d[n-2];
   c[0]   = 0.0;
   c[nm1] = 0.0;
   if (n != 3)
      {
      c[0]   = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0]);
      c[nm1] = c[n-2] / (x[nm1] - x[n-3]) - c[n-3] / (x[n-2] - x[n-4]);
      c[0]   = c[0] * d[0] * d[0] / (x[3] - x[0]);
      c[nm1] = -c[nm1] * d[n-2] * d[n-2] / (x[nm1] - x[n-4]);
      }

   /* Alternative end conditions -- known slopes */
   if (end1 == 1)
      {
      b[0] = 2.0 * (x[1] - x[0]);
      c[0] = (y[1] - y[0]) / (x[1] - x[0]) - slope1;
      }
   if (end2 == 1)
      {
      b[nm1] = 2.0 * (x[nm1] - x[n-2]);
      c[nm1] = slope2 - (y[nm1] - y[n-2]) / (x[nm1] - x[n-2]);
      }

   /* Forward elimination */
   for (i = 1; i < n; ++i)
     {
     t    = d[i-1] / b[i-1];
     b[i] = b[i] - t * d[i-1];
     c[i] = c[i] - t * c[i-1];
     }

   /* Back substitution */
   c[nm1] = c[nm1] / b[nm1];
   for (ib = 0; ib < nm1; ++ib)
      {
      i    = n - ib - 2;
      c[i] = (c[i] - d[i] * c[i+1]) / b[i];
      }

   /* c[i] is now the sigma[i] of the text */

   /* Compute the polynomial coefficients */
   b[nm1] = (y[nm1] - y[n-2]) / d[n-2] + d[n-2] * (c[n-2] + 2.0 * c[nm1]);
   for (i = 0; i < nm1; ++i)
      {
      b[i] = (y[i+1] - y[i]) / d[i] - d[i] * (c[i+1] + 2.0 * c[i]);
      d[i] = (c[i+1] - c[i]) / d[i];
      c[i] = 3.0 * c[i];
      }
   c[nm1] = 3.0 * c[nm1];
   d[nm1] = d[n-2];

   }  /* at least quadratic */

else  /* if n >= 3 */
   {  /* linear segment only  */
   b[0] = (y[1] - y[0]) / (x[1] - x[0]);
   c[0] = 0.0;
   d[0] = 0.0;
   b[1] = b[0];
   c[1] = 0.0;
   d[1] = 0.0;
   }

LeaveSpline:
return 0;
}  /* end of spline() */



//~ double seval (n, u, x, y, b, c, d, last)
//~ int   n;
//~ double u;
//~ double x[], y[], b[], c[], d[];
//~ int    *last;

double seval(const int n, double u, __global double x[], __global double y[], __global double b[], __global double c[], __global double d[], int* last)
//~ __constant double 

{  /* begin function seval() */
	int    i, j, k;
	double w, deltaX;
	
	i = *last;
	if (i >= n-1) i = 0;
	if (i < 0)  i = 0;
	
	if ((x[i] > u) || (x[i+1] < u))
	  {  /* ---- perform a binary search ---- */
	  i = 0;
	  j = n;
	  do
		{
		k = (i + j) / 2;         /* split the domain to search */
		if (u < x[k])  j = k;    /* move the upper bound */
		if (u >= x[k]) i = k;    /* move the lower bound */
		}                        /* there are no more segments to search */
	  while (j > i+1);
	  }
	*last = i;
	
	///* ---- Find closest value (x[i] >= u) ---- */
	//~ int closestIndex;
	//~ int index;
	//~ for(index=0;index<n-1;index++){
		//~ if(x[index]<u && x[index+1]>=u){
			//~ closestIndex = index+1;
			//~ printf("inside 'seval' - closestIndex: %d\n",closestIndex);
			//~ printf("inside 'seval' - x[closestIndex-1]: %f\n",x[closestIndex-1]);
			//~ printf("inside 'seval' - x[closestIndex]: %f\n",x[closestIndex]);
			//~ printf("inside 'seval' - x[closestIndex+1]: %f\n",x[closestIndex+1]);
			//~ printf("inside 'seval' - u: %f\n",u);
		//~ }
	//~ }
	//~ i = closestIndex;
	//~ *last = i;
	
	//~ double t1=0.007, t2=0.003, result=0.0;
	//~ if(x[closestIndex]>u){
		//~ result = 1.0;
	//~ }
	//~ printf("inside 'seval' - result: %f\n",result);
	
	//~ printf("inside 'seval' - closestIndex: %d\n",closestIndex);
	//~ printf("inside 'seval' - closestDistance: %f\n",closestDistance);
	//~ printf("inside 'seval' - x[closestIndex]: %f\n",x[closestIndex]);
	//~ printf("inside 'seval' - y[closestIndex]: %f\n",y[closestIndex]);
	//~ printf("inside 'seval' - closestIndex: %d\n",closestIndex);
	//~ printf("inside 'seval' - x[closestIndex]: %f\n",x[closestIndex]);
		
		
	/* ---- Evaluate the spline ---- */
	deltaX = u - x[i];
	w = y[i] + deltaX * (b[i] + deltaX * (c[i] + deltaX * d[i]));

	/* ---- Debug output ---- */
	//~ printf("inside 'seval' - i: %d\n",i);
	//~ printf("inside 'seval' - x[i]: %f\n",x[i]);
	//~ printf("inside 'seval' - y[i]: %f\n",y[i]);
	//~ printf("inside 'seval' - u: %f\n",u);
	//~ printf("inside 'seval' - deltaX: %f\n",deltaX);
	//~ printf("inside 'seval' - b[i]: %f\n",b[i]);
	//~ printf("inside 'seval' - c[i]: %f\n",c[i]);
	//~ printf("inside 'seval' - d[i]: %f\n",d[i]);
	//~ printf("inside 'seval' - w: %f\n",w);
	//~ printf("inside 'seval' - last: %d\n",*last);

	return (w);
}

__kernel void interpolatePolarCoordinates(__global double* membranePolarRadius,
										  __global double* membranePolarTheta,
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
										  __global double* dbgOut,
										  __global double* b,
										  __global double* c,
										  __global double* d
										 )
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	//~ const int xIndLoc = get_local_id(0);
	//~ const int xSizeLoc = get_local_size(0);
	
	//~ __local double membranePolarThetaLoc[500], membranePolarRadiusLoc[500]; //, interpolationAnglesLoc[500];
	//~ __local double b[500], c[500], d[500];
	
	//~ __local int iflag;
	__private int iflag;
	//~ __local int last;
	
	//~ __private double thetaMaxVal = -10; // initial value larger 2*pi
	//~ __private double thetaMinVal = 10; // initial value smaller -2*pi
	//~ __private int thetaMaxInd; // initial value larger 2*pi
	//~ __private int thetaMinInd; // initial value smaller -2*pi
	
	/*******************************************************************
	 * Order the contour-array from smallest Theta to largest; this is 
	 * needed to perform the interpolation
	 * ****************************************************************/
	//if(xInd==0){
		////~ printf("membranePolarTheta: %f\n",membranePolarTheta[0]);
		//for(int index=0;index<nrOfContourPoints;index++){
			////~ maxIndex = select(maxIndex,index,(maxValue < lineIntensities[index]));
			////~ maxValue = select(maxValue,lineIntensities[index],(maxValue < lineIntensities[index]));
			//if(thetaMaxVal < membranePolarTheta[index]){
				//thetaMaxInd = index;
				//thetaMaxVal = membranePolarTheta[index];
			//}
			//if(thetaMinVal > membranePolarTheta[index]){
				//thetaMinInd = index;
				//thetaMinVal = membranePolarTheta[index];
			//}
			////~ if(maxValue < lineIntensities[index]){
				////~ maxIndex = index;
				////~ maxValue = lineIntensities[index];
			////~ }
			////~ minIndex = select(minIndex,index,(minValue > lineIntensities[index]));
			////~ minValue = select(minValue,lineIntensities[index],(minValue > lineIntensities[index]));
		//}
		////~ printf("thetaMinInd: %d\n",thetaMinInd);
		////~ printf("thetaMaxInd: %d\n",thetaMaxInd);
		////~ printf("xSize: %d\n",xSize);

		//if(thetaMinInd>thetaMaxInd){
			//int counter = 0;
			//for(int index=thetaMinInd;index<nrOfContourPoints;index++){
				//membranePolarThetaLoc[counter] = membranePolarTheta[index];
				//membranePolarRadiusLoc[counter] = membranePolarRadius[index];
				//counter++;
			//}
			//for(int index=0;index<=thetaMaxInd;index++){
				//membranePolarThetaLoc[counter] = membranePolarTheta[index];
				//membranePolarRadiusLoc[counter] = membranePolarRadius[index];
				//counter++;
			//}
		//}
	//}
	
	//~ membranePolarTheta[xInd] = membranePolarThetaLoc[xInd];
	//~ membranePolarRadius[xInd] = membranePolarRadiusLoc[xInd];

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
	/*******************************************************************
	 * write padding array used in the interpolation
	 * ****************************************************************/
	int counter = 0;
	if(xInd==0){
		for(int index=nrOfContourPoints/2;index<nrOfContourPoints;index++){
			interpolationMembranePolarTheta[counter] = membranePolarTheta[index] - 2*3.1415927; // shift abszissa by -2*pi
			interpolationMembranePolarRadius[counter] = membranePolarRadius[index];
			counter++;
		}
		for(int index=0;index<nrOfContourPoints;index++){
			interpolationMembranePolarTheta[counter] = membranePolarTheta[index];
			interpolationMembranePolarRadius[counter] = membranePolarRadius[index];
			counter++;
		}
		for(int index=0;index<nrOfContourPoints/2;index++){
			interpolationMembranePolarTheta[counter] = membranePolarTheta[index] + 2*3.1415927; // shift abszissa by +2*pi
			interpolationMembranePolarRadius[counter] = membranePolarRadius[index];
			counter++;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
	// write values to global array for testing
	if(xInd==0){
		for(int index=0;index<2*nrOfContourPoints;index++){
			interpolationMembranePolarThetaTesting[index] = interpolationMembranePolarTheta[index];
			interpolationMembranePolarRadiusTesting[index] = interpolationMembranePolarRadius[index];
		}
	}
	
	//~ xIndLoc
	if(xInd==0){
		//~ spline (n, end1, end2, slope1, slope2, x, y, b, c, d, iflag)
		
		//~ int returnVal = spline(nrOfInterpolationPoints, 0, 0, 0, 0,
							   //~ membranePolarThetaLoc, membranePolarRadiusLoc,
							   //~ b, c, d, &iflag);
		int returnVal = spline(nrOfInterpolationPoints, 0, 0, 0, 0,
							   interpolationMembranePolarTheta, interpolationMembranePolarRadius,
							   b, c, d, &iflag);

		//~ int ind = 200;
		//~ printf("b[ind]: %f\n",b[ind]);
		//~ printf("c[ind]: %f\n",c[ind]);
		//~ printf("d[ind]: %f\n",d[ind]);
		//~ printf("returnVal: %d\n",returnVal);
		//~ printf("iflag: %d\n",iflag);
		//~ printf("membranePolarThetaLoc[ind]: %f\n",membranePolarThetaLoc[ind]);
		//~ printf("membranePolarRadiusLoc[ind]: %f\n",membranePolarRadiusLoc[ind]);
		
	}

	//~ barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//~ printf("iflag: %d\n",*iflag);
	//~ barrier(CLK_GLOBAL_MEM_FENCE);
	//~ 
	//~ int xIndRef = 1;
	//~ if(xIndLoc==xIndRef){
		//~ printf("xSizeLoc: %d\n",xSizeLoc);
		//~ printf("nrOfInterpolationPoints: %d\n",nrOfInterpolationPoints);
		//~ printf("membranePolarTheta[xIndLoc]: %f\n",membranePolarThetaLoc[xIndLoc]);
		//~ printf("membranePolarRadius[xIndLoc]: %f\n",membranePolarRadiusLoc[xIndLoc]);
		//~ printf("b[xIndLoc]: %f\n",b[xIndLoc]);
		//~ printf("c[xIndLoc]: %f\n",c[xIndLoc]);
		//~ printf("d[xIndLoc]: %f\n",d[xIndLoc]);
		//~ printf("interpolationAnglesLoc[xIndLoc]: %f\n",interpolationAnglesLoc[xIndLoc]);
	//~ }

	//~ dbgOut[xIndLoc] = seval(nrOfInterpolationPoints, interpolationAnglesLoc[xIndLoc], membranePolarThetaLoc, membranePolarRadiusLoc, b, c, d, &last);
	//if(xIndLoc==xIndRef){
		////~ printf("bla");
		//double test = seval(nrOfInterpolationPoints, 1.0, membranePolarThetaLoc, membranePolarRadiusLoc, b, c, d, &last);
		//printf("test : %f\n",test);
		////~ seval(nrOfInterpolationPoints, interpolationAnglesLoc[xIndLoc], membranePolarThetaLoc, membranePolarRadiusLoc, b, c, d, &last);
	//}

	//~ dbgOut[xIndLoc] = seval(nrOfInterpolationPoints, interpolationAnglesLoc[xIndLoc], membranePolarThetaLoc, membranePolarRadiusLoc, b, c, d, &last);
	
	//~ __privat int last = 0.0;
	//~ *last = 0;
	//~ __local int last;
	//~ __private int last = 0;

	__private int last=0;
	
	//~ int index = 303;
	//~ int index = 5;
	//~ int index = 235;
	//~ int index = 0;
	//~ int index = 248;
	//int index = 448;
	//double interpolationAngle = interpolationAngles[index];
	//double output;
	//if(xInd==0){
	   
	   ////~ output = seval(nrOfInterpolationPoints, interpolationAngle,
					   ////~ membranePolarThetaLoc, membranePolarRadiusLoc, b, c, d, &last);
		//output = seval(nrOfInterpolationPoints, interpolationAngle,
					   //interpolationMembranePolarTheta, interpolationMembranePolarRadius, b, c, d, &last);
		
		//printf("#####################################################\n");
		//index = last;
		//printf("last: %d\n",last);
		//printf("interpolationAngle: %f\n",interpolationAngle);
		//printf("output: %f\n",output);
		//printf("interpolationMembranePolarTheta[index-1]: %f\n",interpolationMembranePolarTheta[index-1]);
		//printf("interpolationMembranePolarTheta[index]: %f\n",interpolationMembranePolarTheta[index]);
		//printf("interpolationMembranePolarTheta[index+1]: %f\n",interpolationMembranePolarTheta[index+1]);
		//printf("interpolationMembranePolarRadius[index-1]: %f\n",interpolationMembranePolarRadius[index-1]);
		//printf("interpolationMembranePolarRadius[index]: %f\n",interpolationMembranePolarRadius[index]);
		//printf("interpolationMembranePolarRadius[index+1]: %f\n",interpolationMembranePolarRadius[index+1]);
		////~ printf("membranePolarThetaLoc[index-1]: %f\n",membranePolarThetaLoc[index-1]);
		////~ printf("membranePolarThetaLoc[index]: %f\n",membranePolarThetaLoc[index]);
		////~ printf("membranePolarThetaLoc[index+1]: %f\n",membranePolarThetaLoc[index+1]);
		////~ printf("membranePolarRadiusLoc[index-2]: %f\n",membranePolarRadiusLoc[index-2]);
		////~ printf("membranePolarRadiusLoc[index-1]: %f\n",membranePolarRadiusLoc[index-1]);
		////~ printf("membranePolarRadiusLoc[index]: %f\n",membranePolarRadiusLoc[index]);
		////~ printf("membranePolarRadiusLoc[index+1]: %f\n",membranePolarRadiusLoc[index+1]);
		////~ printf("membranePolarRadiusLoc[index+1]: %f\n",membranePolarRadiusLoc[index+2]);
	//}
	
	if(xInd==0){
		for(int index=0;index<nrOfContourPoints;index++){
			dbgOut[index] = seval(nrOfInterpolationPoints, interpolationAngles[index],
								  interpolationMembranePolarTheta, interpolationMembranePolarRadius, b, c, d, &last);
			//last = last + 1;
			//printf("interpolationAngles[index]: %f\n",interpolationAngles[index]);
			//printf("dbgOut[index]: %f\n",dbgOut[index]);
		}
	}

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

__kernel void findMembranePositionNew(sampler_t sampler, 
									   __read_only image2d_t Img,
									   const int imgSizeX,
									   const int imgSizeY,
									   __constant double* localRotationMatrices, // this should be local or constant
									   __constant double* linFitSearchRangeXvalues, // this should be local or constant
									   const int linFitParameter,
									   __local double* fitIntercept,
									   __local double* fitIncline,
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
									   const int coordinateStartingIndex)
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
	__private double2 rotatedUnitVector;
	rotatedUnitVector.x = localRotationMatrices[4*xIndLoc+0] * membraneNormalVector.x
	                     + localRotationMatrices[4*xIndLoc+1] * membraneNormalVector.y;
	rotatedUnitVector.y = localRotationMatrices[4*xIndLoc+2] * membraneNormalVector.x
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
		Coords.x = basePoint.x + rotatedUnitVector.x * linFitSearchRangeXvalues[index];
		Coords.y = basePoint.y + rotatedUnitVector.y * linFitSearchRangeXvalues[index];
		
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
		
		fit(linFitSearchRangeXvalues, lineIntensities, gradientCenterIndex, linFitParameter, &a, &b, &siga, &sigb, &chi2);
		//~ fitIntercept[xIndLoc] = a;
		fitIntercept[xIndLoc+yIndLoc*xSizeLoc] = a;
		//~ fitIncline[xIndLoc] = b;
		fitIncline[xIndLoc+yIndLoc*xSizeLoc] = b;
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	__private double meanIntensity = 0.0;
	for(int index=0;index<meanParameter;index++){
		Coords.x = basePoint.x + rotatedUnitVector.x * ( meanRangeXvalues[index] + meanRangePositionOffset );
		Coords.y = basePoint.y + rotatedUnitVector.y * ( meanRangeXvalues[index] + meanRangePositionOffset );
		
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
	relativeMembranePositionLocalCoordSys = (meanIntensity-fitIntercept[xIndLoc+yIndLoc*xSizeLoc])/fitIncline[xIndLoc+yIndLoc*xSizeLoc];
	
	//~ localMembranePositionsX[xIndLoc] = basePoint.x + rotatedUnitVector.x * relativeMembranePositionLocalCoordSys;
	localMembranePositionsX[xIndLoc+yIndLoc*xSizeLoc] = basePoint.x + rotatedUnitVector.x * relativeMembranePositionLocalCoordSys;
	//~ localMembranePositionsY[xIndLoc] = basePoint.y + rotatedUnitVector.y * relativeMembranePositionLocalCoordSys;
	localMembranePositionsY[xIndLoc+yIndLoc*xSizeLoc] = basePoint.y + rotatedUnitVector.y * relativeMembranePositionLocalCoordSys;
	
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
	
	__private double xTmp = 0.0, yTmp = 0.0, meanIncline = 0.0;
	//~ [xIndLoc+yIndLoc*xSizeLoc]
	if(xIndLoc==0){
		//~ for(int yIndLoc=get_local_id(0);yIndLoc++;yIndLoc<ySizeLoc){
			//~ for(int index=get_local_id(1);index<xSize;index++){
			for(int index=0;index<xSize;index++){
				//~ xTmp += fitIncline[index] * localMembranePositionsX[index];
				xTmp += fitIncline[index+yIndLoc*xSizeLoc] * localMembranePositionsX[index+yIndLoc*xSizeLoc];
				//~ yTmp += fitIncline[index] * localMembranePositionsY[index];
				yTmp += fitIncline[index+yIndLoc*xSizeLoc] * localMembranePositionsY[index+yIndLoc*xSizeLoc];
				//~ meanIncline += fitIncline[index];
				meanIncline += fitIncline[index+yIndLoc*xSizeLoc];
			}
			membraneCoordinatesX[coordinateIndex] = xTmp/meanIncline;
			membraneCoordinatesY[coordinateIndex] = yTmp/meanIncline;

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

__kernel void sortPolarCoordinates(__global double* membranePolarRadius,
								  __global double* membranePolarTheta,
								  //~ __local double* membranePolarRadiusLoc,
								  //~ __local double* membranePolarThetaLoc,
								  //~ __local double* membranePolarRadiusLoc,
								  //~ __local double* membranePolarThetaLoc,
								  const int nrOfContourPoints
								  )
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
					__private double tmp1 = membranePolarTheta[i-1];
					membranePolarTheta[i-1] = membranePolarTheta[i];
					membranePolarTheta[i] = tmp1;
					__private double tmp2 = membranePolarRadius[i-1];
					membranePolarRadius[i-1] = membranePolarRadius[i];
					membranePolarRadius[i] = tmp2;
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
	
	//~ procedure bubbleSort( A : list of sortable items )
		//~ n = length(A)
		//~ repeat
		   //~ newn = 0
		   //~ for i = 1 to n-1 inclusive do
			  //~ if A[i-1] > A[i] then
				 //~ swap(A[i-1], A[i])
				 //~ newn = i
			  //~ end if
		   //~ end for
		   //~ n = newn
		//~ until n = 0
	//~ end procedure

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
__kernel void findMembranePositionNew2(sampler_t sampler, 
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
		
		fit(linFitSearchRangeXvalues, lineIntensities, gradientCenterIndex, linFitParameter, &a, &b, &siga, &sigb, &chi2);
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

//__kernel void filterLargeDsValues(__global double* membraneCoordinatesX,
								  //__global double* membraneCoordinatesY,
								  //__global double* membraneNormalVectorsX,
								  //__global double* membraneNormalVectorsY,
								  //__global int* closestLowerAcceptableValueIndex,
								  //__global int* closestUpperAcceptableValueIndex,
								  //__local int* closestLowerAcceptableValueIndexLoc,
								  //__local int* closestUpperAcceptableValueIndexLoc,
								  //double tolerance, // distance an pixel that a point is allowed to be off (expected to be ~0.5...1 pixels)
								  //__global double* ds
								//)
//{
	//const int xInd = get_global_id(0);
	//const int xSize = get_global_size(0);

	//__private bool largeValueLeft;
	////~ __local int closestLowerAcceptableValueIndexLoc[2048];
	////~ __local int closestUpperAcceptableValueIndexLoc[2048];
	
	//closestLowerAcceptableValueIndexLoc[xInd] = closestLowerAcceptableValueIndex[xInd];
	//closestUpperAcceptableValueIndexLoc[xInd] = closestUpperAcceptableValueIndex[xInd];
	
	////~ if(xInd>1998&xInd<2002){
		////~ printf("closestLowerAcceptableValueIndex[xInd], before: %d\n",closestLowerAcceptableValueIndexLoc[xInd]);
		////~ printf("closestUpperAcceptableValueIndex[xInd], before: %d\n",closestUpperAcceptableValueIndexLoc[xInd]);
	////~ }
	
	//do{
		//largeValueLeft = false;
		//if(ds[closestLowerAcceptableValueIndexLoc[xInd]]>tolerance){
			//closestLowerAcceptableValueIndexLoc[xInd] -= 1;
			//largeValueLeft = true;
		//}

		//if(membraneCoordinatesX[closestUpperAcceptableValueIndexLoc[xInd]]>tolerance){
			//closestUpperAcceptableValueIndexLoc[xInd] += 1;
			//largeValueLeft = true;
		//}
		////~ id = getGlobalId();
		////~ output[id] = input[id] * input[id];
		//if(closestLowerAcceptableValueIndexLoc[xInd]<0){ // avoid that we round out array bounds by using periodic boundaries
			//closestLowerAcceptableValueIndexLoc[xInd] = closestLowerAcceptableValueIndexLoc[xInd]+xSize;
		//}
		//if(closestUpperAcceptableValueIndexLoc[xInd]>xSize-1){ // avoid that we round out array bounds by using periodic boundaries
			//closestUpperAcceptableValueIndexLoc[xInd] = closestUpperAcceptableValueIndexLoc[xInd]-xSize;
		//}
	//}while(largeValueLeft);
	
	////~ if(xInd>1998&xInd<2002){
		////~ printf("closestLowerAcceptableValueIndex[xInd], after: %d\n",closestLowerAcceptableValueIndexLoc[xInd]);
		////~ printf("closestUpperAcceptableValueIndex[xInd], after: %d\n",closestUpperAcceptableValueIndexLoc[xInd]);
	////~ }

	//* *****************************************************************
	 //* interpolate locations that are NaN 
	 //* ****************************************************************/
	//membraneCoordinatesX[xInd] = (membraneCoordinatesX[closestLowerAcceptableValueIndexLoc[xInd]] + membraneCoordinatesX[closestUpperAcceptableValueIndexLoc[xInd]])/2;
	//membraneCoordinatesY[xInd] = (membraneCoordinatesY[closestLowerAcceptableValueIndexLoc[xInd]] + membraneCoordinatesY[closestUpperAcceptableValueIndexLoc[xInd]])/2;
	
	//membraneNormalVectorsX[xInd] = (membraneNormalVectorsX[closestLowerAcceptableValueIndexLoc[xInd]] + membraneNormalVectorsX[closestUpperAcceptableValueIndexLoc[xInd]])/2;
	//membraneNormalVectorsY[xInd] = (membraneNormalVectorsY[closestLowerAcceptableValueIndexLoc[xInd]] + membraneNormalVectorsY[closestUpperAcceptableValueIndexLoc[xInd]])/2;

	////~ membraneNormalNorm = sqrt( pow(xMembraneNormalTmp,2) + pow(yMembraneNormalTmp,2) );
	
	//double membraneNormalNorm;
	//membraneNormalNorm = sqrt( pow(membraneNormalVectorsX[xInd],2) + pow(membraneNormalVectorsY[xInd],2) );
	
	//membraneNormalVectorsX[xInd] = membraneNormalVectorsX[xInd]/membraneNormalNorm;
	//membraneNormalVectorsY[xInd] = membraneNormalVectorsY[xInd]/membraneNormalNorm;

//}


__kernel void findMembranePositionNew3(sampler_t sampler, 
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
									   const int coordinateStartingIndex,
									   const double inclineTolerance,
									   const double meanIntensity
									   )
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
		
		fit(linFitSearchRangeXvalues, lineIntensities, gradientCenterIndex, linFitParameter, &a, &b, &siga, &sigb, &chi2);
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
	
	//__private double meanIntensity = 0.0;
	//for(int index=0;index<meanParameter;index++){
		////~ Coords.x = basePoint.x + rotatedUnitVector.x * ( meanRangeXvalues[index] + meanRangePositionOffset );
		////~ Coords.y = basePoint.y + rotatedUnitVector.y * ( meanRangeXvalues[index] + meanRangePositionOffset );
		//Coords.x = basePoint.x + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].x * ( meanRangeXvalues[index] + meanRangePositionOffset );
		//Coords.y = basePoint.y + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].y * ( meanRangeXvalues[index] + meanRangePositionOffset );
		
		//NormCoords = Coords/convert_float2(dims);
		
		//meanIntensity = meanIntensity + read_imagef(Img, sampler, NormCoords).x;
	//}
	//meanIntensity = meanIntensity/convert_float(meanParameter);
	
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


__kernel void filterPolarCoordinateSingularities_OLD(__global double* membranePolarRadius,
												 __global double* membranePolarTheta,
												 const double minAngleDifference
												 //~ __global double* dbgOut,
												 //~ __global double* dbgOut2
												 )
{
	//~ A · B = A B cos θ = |A||B| cos θ

	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	//~ __private double2 basePoint = {membraneCoordinatesX[coordinateIndex],membraneCoordinatesY[coordinateIndex]};
	
	//~ __private double minAngleDifference = 1e-3;
	
	if(xInd>0 && xInd<xSize-3){ // calculate interior gradients
		if(fabs(membranePolarTheta[xInd] - membranePolarTheta[xInd+1]) < minAngleDifference){
			__private double meanAngle = (membranePolarTheta[xInd] + membranePolarTheta[xInd+1])/2;
			__private double meanRadius = (membranePolarRadius[xInd] + membranePolarRadius[xInd+1])/2;
			
			membranePolarTheta[xInd] = (meanAngle + membranePolarTheta[xInd-1])/2;
			membranePolarTheta[xInd+1] = (meanAngle + membranePolarTheta[xInd+2])/2;

			membranePolarRadius[xInd] = (meanRadius + membranePolarRadius[xInd-1])/2;
			membranePolarRadius[xInd+1] = (meanRadius + membranePolarRadius[xInd+2])/2;
		}
	}
	else if(xInd==xSize-2){ // calculate edge gradient
		if(fabs(membranePolarTheta[xInd] - membranePolarTheta[xInd+1]) < minAngleDifference){
			__private double meanAngle = (membranePolarTheta[xInd] + membranePolarTheta[xInd+1])/2;
			__private double meanRadius = (membranePolarRadius[xInd] + membranePolarRadius[xInd+1])/2;
			
			membranePolarTheta[xInd] = (meanAngle + membranePolarTheta[xInd-1])/2;
			membranePolarTheta[xInd+1] = (meanAngle + membranePolarTheta[0])/2;

			membranePolarRadius[xInd] = (meanRadius + membranePolarRadius[xInd-1])/2;
			membranePolarRadius[xInd+1] = (meanRadius + membranePolarRadius[0])/2;
		}
	}
	else if(xInd==xSize-1){ // calculate edge gradient
		if(fabs(membranePolarTheta[xInd] - membranePolarTheta[0]) < minAngleDifference){
			__private double meanAngle = (membranePolarTheta[xInd] + membranePolarTheta[0])/2;
			__private double meanRadius = (membranePolarRadius[xInd] + membranePolarRadius[0])/2;
			
			membranePolarTheta[xInd] = (meanAngle + membranePolarTheta[xInd-1])/2;
			membranePolarTheta[0] = (meanAngle + membranePolarTheta[1])/2;

			membranePolarRadius[xInd] = (meanRadius + membranePolarRadius[xInd-1])/2;
			membranePolarRadius[0] = (meanRadius + membranePolarRadius[1])/2;
		}
	}
	else if(xInd==0){ // calculate edge gradient
		if(fabs(membranePolarTheta[xInd] - membranePolarTheta[xInd+1]) < minAngleDifference){
			__private double meanAngle = (membranePolarTheta[xInd] + membranePolarTheta[xInd+1])/2;
			__private double meanRadius = (membranePolarRadius[xInd] + membranePolarRadius[xInd+1])/2;
			
			membranePolarTheta[xInd] = (meanAngle + membranePolarTheta[xSize-1])/2;
			membranePolarTheta[xInd+1] = (meanAngle + membranePolarTheta[xInd+2])/2;

			membranePolarRadius[xInd] = (meanRadius + membranePolarRadius[xSize-1])/2;
			membranePolarRadius[xInd+1] = (meanRadius + membranePolarRadius[xInd+2])/2;
		}
	}
	
}

__kernel void calculateAngleDifference( __global double* membranePolarTheta,
										__global double* angleDifferencesUpper,
										__global double* angleDifferencesLower
										)
{
	const int xInd = get_global_id(0);
	const int xSize = get_global_size(0);
	
	if(xInd > 0 && xInd < xSize-1){
		angleDifferencesUpper[xInd] = fabs(membranePolarTheta[xInd+1] - membranePolarTheta[xInd]);
		angleDifferencesLower[xInd] = fabs(membranePolarTheta[xInd] - membranePolarTheta[xInd-1]);
	}
	else if(xInd == 0){
		angleDifferencesUpper[xInd] = fabs(membranePolarTheta[xInd+1] - membranePolarTheta[xInd]);
		angleDifferencesLower[xInd] = fabs(membranePolarTheta[xInd] - membranePolarTheta[xSize-1]);
	}
	else if(xInd == xSize-1){
		angleDifferencesUpper[xInd] = fabs(membranePolarTheta[0] - membranePolarTheta[xInd]);
		angleDifferencesLower[xInd] = fabs(membranePolarTheta[xInd] - membranePolarTheta[xInd-1]);
	}
}

__kernel void filterPolarCoordinateSingularities(__global double* membranePolarRadius,
												 __global double* membranePolarTheta,
												 __global double* angleDifferencesUpper,
												 __global double* angleDifferencesLower,
												 __global double2* previousContourCenter,
												 //~ __global double* interCoordinateAngles,
												 __global double* membraneCoordinatesX,
												 __global double* membraneCoordinatesY,
												 __global double* membraneNormalVectorsX,
												 __global double* membraneNormalVectorsY,
												 __local int* closestLowerCorrectIndexLoc,
												 __local int* closestUpperCorrectIndexLoc,
												 const double minAngleDifference
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
	
	//~ __private double angleDifference = fabs(membranePolarTheta[xInd] - membranePolarTheta[xInd+1]);
	
	//~ (fabs(membranePolarTheta[xInd] - membranePolarTheta[xInd+1]) < minAngleDifference)
	
	//~ if(xInd>1998&xInd<2002){
		//~ printf("closestLowerCorrectIndex[xInd], before: %d\n",closestLowerCorrectIndexLoc[xInd]);
		//~ printf("closestUpperCorrectIndex[xInd], before: %d\n",closestUpperCorrectIndexLoc[xInd]);
	//~ }
	
	__private int distToLowerIndex = 0;
	__private int distToUpperIndex = 0;
	do{
		incorrectValueLeft = false;
		//~ if(isnan(membraneCoordinatesX[closestLowerCorrectIndexLoc[xInd]])){
		//~ if(interCoordinateAngles[closestLowerCorrectIndexLoc[xInd]] > maxInterCoordinateAngle){
		if(angleDifferencesUpper[closestLowerCorrectIndexLoc[xInd]] < minAngleDifference ||
		   angleDifferencesLower[closestLowerCorrectIndexLoc[xInd]] < minAngleDifference ){
			closestLowerCorrectIndexLoc[xInd] -= 1;
			distToLowerIndex++;
			incorrectValueLeft = true;
		}
		
		//~ if(isnan(membraneCoordinatesX[closestUpperCorrectIndexLoc[xInd]])){
		if(angleDifferencesUpper[closestUpperCorrectIndexLoc[xInd]] < minAngleDifference ||
		   angleDifferencesLower[closestUpperCorrectIndexLoc[xInd]] < minAngleDifference){
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
		membranePolarRadius[xInd] = ((double)distToLowerIndex * membranePolarRadius[closestLowerCorrectIndexLoc[xInd]] 
								   + (double)distToUpperIndex * membranePolarRadius[closestUpperCorrectIndexLoc[xInd]])
									/(double)(distToLowerIndex+distToUpperIndex);
		//~ membraneCoordinatesX[xInd] = ((double)distToLowerIndex * membraneCoordinatesX[closestLowerCorrectIndexLoc[xInd]] 
									//~ + (double)distToUpperIndex * membraneCoordinatesX[closestUpperCorrectIndexLoc[xInd]])
									//~ /(double)(distToLowerIndex+distToUpperIndex);
		
		membranePolarTheta[xInd] = ((double)distToLowerIndex * membranePolarTheta[closestLowerCorrectIndexLoc[xInd]] 
								  + (double)distToUpperIndex * membranePolarTheta[closestUpperCorrectIndexLoc[xInd]])
								   /(double)(distToLowerIndex+distToUpperIndex);
		//~ membraneCoordinatesY[xInd] = ((double)distToLowerIndex * membraneCoordinatesY[closestLowerCorrectIndexLoc[xInd]] 
									//~ + (double)distToUpperIndex * membraneCoordinatesY[closestUpperCorrectIndexLoc[xInd]])
									//~ /(double)(distToLowerIndex+distToUpperIndex);
		
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
	
	// radialLineDirectionVector <-> b
	// radialLineBasePoint <-> c
	__private double2 radialLineBasePoint = contourCenter[0];
	__private double2 radialLineDirectionVector = radialVectors[xInd];
	
	__private double distanceFromCenter = 160000;
	__private double2 interpolatedMembranePoint;
	__private double2 radialVector;
		
	for(index=xInd-nrOfAnglesToCompare/2+1;index<xInd+nrOfAnglesToCompare/2;index++)
	{
		int lowerIndex = index;
		int upperIndex = index+1;
		
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
		
		__private double lowerAngle = membranePolarTheta[lowerIndex];
		__private double upperAngle = membranePolarTheta[upperIndex];
		__private double interpolationAngle = interpolationAngles[xInd];
		
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

			//
			
			//~ if(xInd==1536){
				//~ printf("radialLineBasePoint.x: %f\n",radialLineBasePoint.x);
				//~ printf("radialLineBasePoint.y: %f\n",radialLineBasePoint.y);
				//~ printf("radialLineDirectionVector.x: %f\n",radialLineDirectionVector.x);
				//~ printf("radialLineDirectionVector.y: %f\n",radialLineDirectionVector.y);
				//~ printf("\n");
				//~ printf("lineSegmentBasePoint.x: %f\n",lineSegmentBasePoint.x);
				//~ printf("lineSegmentBasePoint.y: %f\n",lineSegmentBasePoint.y);
				//~ printf("lineSegmentDirectionVector.x: %f\n",lineSegmentDirectionVector.x);
				//~ printf("lineSegmentDirectionVector.y: %f\n",lineSegmentDirectionVector.y);
				//~ printf("\n");
				//~ printf("membraneCoord[lowerIndex]: %f,%f\n",membraneCoordinatesX[lowerIndex],membraneCoordinatesY[lowerIndex]);
				//~ printf("membraneCoord[upperIndex]: %f,%f\n",membraneCoordinatesX[upperIndex],membraneCoordinatesY[upperIndex]);
				//~ printf("interceptPoint: %f,%f\n",interceptPoint.x,interceptPoint.y);
			//~ }
		}
	}
	interpolatedMembraneCoordinatesX[xInd] = interpolatedMembranePoint.x;
	interpolatedMembraneCoordinatesY[xInd] = interpolatedMembranePoint.y;
}
