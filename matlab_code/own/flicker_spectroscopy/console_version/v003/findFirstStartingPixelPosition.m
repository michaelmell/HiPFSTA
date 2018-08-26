function startingPixelPosition = findFirstStartingPixelPosition(imageData,center,trackingParameters,trackingVariables)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% findStartingPixelPosition   Function for finding the starting pixel 
%%% position for the contour tracking algorithm; the pixel position is 
%%% searched in vertical direction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

startingPixelPosition = zeros(1,2);
% chosen y-pixelpostion for finding membrane
startingPixelPosition(1,2) = round(center(2));

% determine the x-postion of starting pixel for chosen y-positions
% startingPixelPosition(1,1) = pixelposition(imageData(:,round(center(2))),round(center(1)-trackingVariables.sizeofvesicle-30),...
%                                     round(center(1)),trackingParameters.linfitparameter);
startingPixelPosition(1,1) = pixelposition(imageData(:,round(center(2))),round(center(1)-trackingVariables.sizeofvesicle),...
                                    round(center(1)),trackingParameters.linfitparameter);