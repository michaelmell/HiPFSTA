function [xymempos,pixpos,contourclosed] = contourTrack(imageData,startingPixelPosition,trackingParameters,trackingVariables)
try
% for first point set y membrane position to pixel position, since not better known
trackingVariables.pixpos(1,:) = startingPixelPosition;
trackingVariables.xymempos(1,:) = startingPixelPosition;

% coordinate transformation to the vw-coordinate system
trackingVariables.vwmempos(1,1) = trackingVariables.xymempos(1,1) * 0.70710678 + trackingVariables.xymempos(1,2) * 0.70710678; % trafo to v-axis
trackingVariables.vwmempos(1,2) = trackingVariables.xymempos(1,2) * 0.70710678 - trackingVariables.xymempos(1,1) * 0.70710678; % trafo to w-axis

% set control variables
trackingVariables.skipImage = 0;

for index2 = 1:trackingParameters.maxnrofpoints
    % this serves to count the number of detection iterations after the last
    % directionchange
    trackingVariables.xDirectionChangeCounter = trackingVariables.xDirectionChangeCounter+1;
    trackingVariables.yDirectionChangeCounter = trackingVariables.yDirectionChangeCounter+1;
    trackingVariables.vDirectionChangeCounter = trackingVariables.vDirectionChangeCounter+1;
    trackingVariables.wDirectionChangeCounter = trackingVariables.wDirectionChangeCounter+1;

    % after the first number of points given by the variable
    % 'directiondetectionstart' the following conditions decide over a possible 
    % direction change in x- and y-direction
    trackingVariables = checkDirectionChange(index2,trackingParameters,trackingVariables);

    % calculate new membrane position
    trackingVariables = calcNewMembranePosition(imageData,index2,trackingParameters,trackingVariables);

    % calculate new pixel position
    trackingVariables = calcNewPixelPosition(imageData,index2,trackingParameters,trackingVariables);
    
    trackingVariables = checkThatNewPixelInsideImage(index2,trackingVariables,trackingParameters);
    if trackingVariables.skipImage == 1
        break;
    end

    % check break-off condition
    [contourclosed,breakofpixel] = checkBreakOffCondition(index2,trackingParameters,trackingVariables);

    % write to the variable 'trackingVariables.contourclosedindex' whether the contour was closed
    if contourclosed == 1
        break;
    end
end

% remove the duplicately tracked membrane positions
trackingVariables.xymempos = trackingVariables.xymempos(breakofpixel:index2-1,:);
trackingVariables.pixpos = trackingVariables.pixpos(breakofpixel:index2-1,:);

% remove trailing zeros from coordinate-vectors
trackingVariables.xymempos(trackingVariables.xymempos(:,1)==0,:) = [];
trackingVariables.pixpos(trackingVariables.pixpos(:,1)==0,:) = [];

% set output
xymempos = trackingVariables.xymempos;
pixpos = trackingVariables.pixpos;
catch exception
	display('###################### An error occured! ######################')
    display(getReport(exception));
    keyboard 
end