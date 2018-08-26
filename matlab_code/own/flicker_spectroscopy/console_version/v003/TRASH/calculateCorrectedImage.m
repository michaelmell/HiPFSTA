function parameterStruct = calculateCorrectedImage(parameterStruct)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% calculateCorrectedImage correct the image with the background
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

correctedImage = parameterStruct.image;

% do background correction if requested
if parameterStruct.do_background_correction == 1
    background = parameterStruct.background;
    correctedImage = correctedImage./background;
end

% do filtering if requested
if parameterStruct.do_gaussian_filtering == 1
    correctedImage = filterWithGaussian(correctedImage, ...
                                        parameterStruct.gaussian_filter_width);
end

parameterStruct.correctedImage = correctedImage;