function trackingVariables = checkThatNewPixelInsideImage(index2,trackingVariables,trackingParameters)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% checkThatNewPixelInsideImage    This function checks that the next 
%%% pixel and the surrounding area for the calculations is still inside the
%%% image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% trackingVariables.pixpos(index2+1,1)+trackingVariables.xdirection,
% trackingVariables.pixpos(index2+1,2)-trackingParameters.meanparameter > 0
% trackingVariables.pixpos(index2+1,2)+trackingParameters.meanparameter < trackingVariables.imageymax
% trackingVariables.pixpos(index2+1,1)-trackingParameters.meanparameter > 0
% trackingVariables.pixpos(index2+1,1)+trackingParameters.meanparameter < trackingVariables.imagexmax
% trackingVariables.pixpos(index2+1,2)+trackingVariables.ydirection

if ~( trackingVariables.pixpos(index2+1,2)-trackingParameters.meanparameter > 0 && ...
      trackingVariables.pixpos(index2+1,2)+trackingParameters.meanparameter < trackingVariables.imageymax && ...
      trackingVariables.pixpos(index2+1,1)-trackingParameters.meanparameter > 0 && ...
      trackingVariables.pixpos(index2+1,1)+trackingParameters.meanparameter < trackingVariables.imagexmax )
    display('pixelposition or surrounding area for calculations is out-of-bounds (too large or to small)');
    trackingVariables.outsidecounter = trackingVariables.outsidecounter + 1;
    trackingVariables.skipImage = 1;
end

% if trackingVariables.pixpos(index2+1,1) - trackingParameters.meanparameter - 1 < 0 || ...
%    trackingVariables.pixpos(index2+1,1) + trackingParameters.meanparameter + 1 > trackingVariables.imagexmax || ...
%    trackingVariables.pixpos(index2+1,2) - trackingParameters.meanparameter - 1 < 0 || ...
%    trackingVariables.pixpos(index2+1,1) + trackingParameters.meanparameter + 1 > trackingVariables.imageymax
%     display('pixelposition or surrounding area for calculations is out-of-bounds (too large or to small)');
%     trackingVariables.outsidecounter = trackingVariables.outsidecounter + 1;
%     trackingVariables.skipImage = 1;
% end
