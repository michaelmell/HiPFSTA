function [directionChangeCounter,direction] = checkAndSetNewDirection(directionChangeCounter,direction,meanMembraneCoordinateChange,trackingParameters)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% checkAndSetNewDirection     Function which checks whether the direction 
%%% of the tracking algorithm should be changed and changes it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% the code for this IF-query does the same as below, but is faster -
% although less expressiv
if directionChangeCounter > trackingParameters.directionchange && meanMembraneCoordinateChange < trackingParameters.directioncondition
        direction = - direction;
        directionChangeCounter = 0;
end

% if directionChangeCounter > trackingParameters.directionchange
%     if meanMembraneCoordinateChange < trackingParameters.directioncondition
%         direction = - direction;
%         directionChangeCounter = 0;
%     end
% end