function trackingVariables = testContour(contourNrIndex,trackingVariables,trackingParameters)

% if contour was closed add it as closed to the index
if trackingVariables.contourclosed == 1
    trackingVariables.contourclosedindex(contourNrIndex) = 1;
    trackingVariables.contourclosedcounter = trackingVariables.contourclosedcounter + 1;
end

% Calculate new center position of vesicle in case the vesicle is moving;
% this is done after the number of detection iteration steps given by the 
% parameter "newcentersteps"
trackingVariables.centercounter = trackingVariables.centercounter + 1;
[newcenter,contourCircumference,contourRadius] = calcContourProperties(trackingVariables.xymempos);
trackingVariables.contourRadius = contourRadius; % TODO: REMOVE THIS LINE
setNewCenter = checkNewCenterConditions(trackingVariables,trackingParameters,newcenter);

if setNewCenter
    trackingVariables.center = newcenter;
end