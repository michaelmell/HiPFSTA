function result = checkPixelIndexSanity( trackingVariables )

keyboard

% check that new pixel index is within image in x-direction
trackingVariables.pixpos(index2,1)+trackingVariables.xdirection,
trackingVariables.pixpos(index2,2)-trackingParameters.meanparameter,
trackingVariables.pixpos(index2,2)+trackingParameters.meanparameter

trackingVariables.pixpos(index2+1,1)+trackingVariables.xdirection,
trackingVariables.pixpos(index2+1,2)-trackingParameters.meanparameter,
trackingVariables.pixpos(index2+1,2)+trackingParameters.meanparameter

% check that new pixel index is within image in y-direction
trackingVariables.pixpos(index2,1)-trackingParameters.meanparameter,
trackingVariables.pixpos(index2,1)+trackingParameters.meanparameter,
trackingVariables.pixpos(index2,2)+trackingVariables.ydirection

trackingVariables.pixpos(index2+1,1)-trackingParameters.meanparameter,
trackingVariables.pixpos(index2+1,1)+trackingParameters.meanparameter,
trackingVariables.pixpos(index2+1,2)+trackingVariables.ydirection
