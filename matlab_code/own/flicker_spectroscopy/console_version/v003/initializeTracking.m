function [programParameters,trackingParameters,trackingVariables] = initializeTracking(parameterStruct)

% set program parameters
programParameters = setProgramParameters(parameterStruct);

% get number of images to be analyzed
programParameters = getNumberOfImages(programParameters);

% set tracking parameters
trackingParameters = parameterStruct;
trackingVariables = setTrackingVariables(trackingParameters,programParameters);

% set dynamicaly adjusted parameters
% calculate the the corresponding parameters for the diagonal calculations
trackingParameters.diagLinFitParameter = ceil(trackingParameters.linfitparameter/1.4142);
trackingParameters.diagMeanParameter = ceil(trackingParameters.meanparameter/1.4142);
trackingParameters.displaystartcounter = trackingParameters.displaystart;
