function void = trackImages(configFilePath)

parameterStruct = loadToStructure(configFilePath);

% parameterStruct = handles.parameterStruct;
% [programParameters,trackingParameters,trackingVariables] = initializeTracking(parameterStruct);
% handles.trackingVariables = trackingVariables;
% % setup up file system information
% programParameters = getNumberOfImages(programParameters);

% trackingVariables.contourRadius = handles.trackingVariables.contourRadius;
% trackingVariables.startingPixelPosition = handles.trackingVariables.startingPixelPosition;
% trackingVariables = handles.trackingVariables;
% keyboard

console_tracking(parameterStruct);
