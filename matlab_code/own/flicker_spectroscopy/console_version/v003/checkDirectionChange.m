function trackingVariables = checkDirectionChange(index2,trackingParameters,trackingVariables)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% checkDirectionChange    This function checks the whether the conditions
%%% for a direction change are met and changes it; it does this by calling
%%% function 'checkAndSetNewDirection.m'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% after the first number of points given by the variable
% 'directiondetectionstart' the following conditions decide over a possible 
% direction change in x- and y-direction

if index2 > trackingParameters.directiondetectionstart 
    trackingVariables.xymeanmemchange(index2,:) = abs([(meanOneDimension(trackingVariables.xymempos(index2-trackingParameters.lastpositions:index2,1)-trackingVariables.xymempos(index2-trackingParameters.lastpositions,1))), ...
                                                       (meanOneDimension(trackingVariables.xymempos(index2-trackingParameters.lastpositions:index2,2)-trackingVariables.xymempos(index2-trackingParameters.lastpositions,2)))]);

    trackingVariables.vwmeanmemchange(index2,:) = abs([(meanOneDimension(trackingVariables.vwmempos(index2-trackingParameters.lastpositions:index2,1)-trackingVariables.vwmempos(index2-trackingParameters.lastpositions,1))), ...
                                                       (meanOneDimension(trackingVariables.vwmempos(index2-trackingParameters.lastpositions:index2,2)-trackingVariables.vwmempos(index2-trackingParameters.lastpositions,2)))]);

    % set new x-direction
    [trackingVariables.xDirectionChangeCounter,trackingVariables.xdirection] = ...
        checkAndSetNewDirection(trackingVariables.xDirectionChangeCounter,trackingVariables.xdirection,trackingVariables.xymeanmemchange(index2,1),trackingParameters);

    % set new y-direction
    [trackingVariables.yDirectionChangeCounter,trackingVariables.ydirection] = ...
        checkAndSetNewDirection(trackingVariables.yDirectionChangeCounter,trackingVariables.ydirection,trackingVariables.xymeanmemchange(index2,2),trackingParameters);

    % set new v-direction
    [trackingVariables.vDirectionChangeCounter,trackingVariables.vdirection] = ...
        checkAndSetNewDirection(trackingVariables.vDirectionChangeCounter,trackingVariables.vdirection,trackingVariables.vwmeanmemchange(index2,1),trackingParameters);
    
    %set new w-direction
    [trackingVariables.wDirectionChangeCounter,trackingVariables.wdirection] = ...
        checkAndSetNewDirection(trackingVariables.wDirectionChangeCounter,trackingVariables.wdirection,trackingVariables.vwmeanmemchange(index2,2),trackingParameters);
end