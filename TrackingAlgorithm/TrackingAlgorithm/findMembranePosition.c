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
								   const double inclineTolerance
								   )
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
	
	__private double lineIntensities[400];
	
	__private double2 membraneNormalVector = membraneNormalVectors[coordinateIndex];
	
	// matrix multiplication with linear array of sequential 2x2 rotation matrices
	rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].x = localRotationMatrices[4*xIndLoc+0] * membraneNormalVector.x
												   + localRotationMatrices[4*xIndLoc+1] * membraneNormalVector.y;
	rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc].y = localRotationMatrices[4*xIndLoc+2] * membraneNormalVector.x
												   + localRotationMatrices[4*xIndLoc+3] * membraneNormalVector.y;

	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	__private int maxIndex;
	__private int minIndex;
	__private double minValue = 32000; // initialize value with first value of array
	__private double maxValue = 0; // initialize value with first value of array
	
	__private double2 Coords;
	__private double2 NormCoords;
	__private const int2 dims = get_image_dim(Img);
	
	__private double2 basePoint = membraneCoordinates[coordinateIndex];
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int index=0;index<imgSizeY;index++) // TODO: The maximum index range 'imgSizeY' is almost certainly wrong here! It should run till the max length of 'linFitSearchRangeXvalues'. - Michael 2017-04-16
	{
		Coords = basePoint + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc] * linFitSearchRangeXvalues[index];
		
		NormCoords = Coords/convert_double2(dims);
		
		float2 fNormCoords;
		fNormCoords = convert_float2(NormCoords);
		
		lineIntensities[index] = read_imagef(Img, sampler, fNormCoords)[0];
	}
	
	for(int index=0;index<imgSizeY;index++) // TODO: The maximum index range 'imgSizeY' is almost certainly wrong here! It should run till the max length of 'linFitSearchRangeXvalues'. - Michael 2017-04-16
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
		fitIntercept[xIndLoc+yIndLoc*xSizeLoc] = a;
		fitIncline[xIndLoc+yIndLoc*xSizeLoc] = b;
	}
	else
	{
		fitIntercept[xIndLoc+yIndLoc*xSizeLoc] = 0; // so that later they are not counted in the weighted sum (see below...)
		fitIncline[xIndLoc+yIndLoc*xSizeLoc] = 0;
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	__private double meanIntensity = 0.0;
	for(int index=0;index<meanParameter;index++)
	{
		Coords = basePoint + rotatedUnitVector2[xIndLoc+yIndLoc*xSizeLoc] * ( meanRangeXvalues[index] + meanRangePositionOffset );
		
		NormCoords = Coords/convert_double2(dims);
		
		float2 fNormCoords;
		fNormCoords = convert_float2(NormCoords);

		meanIntensity = meanIntensity + read_imagef(Img, sampler, fNormCoords)[0];
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


