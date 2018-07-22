function setNewCenter = checkNewCenterConditions(trackingVariables,trackingParameters,newcenter)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% checkNewCenterConditions    Function to check whether the conditions
%%% for setting new center coordinates were met
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

setNewCenter = 0;

if trackingVariables.contourclosed == 1
        trackingVariables.centercounter = 0;        
        % this query serves to prevent the center from making to big a jump from
        % one contour to the next as may happen when the algorithm didn't track
        % the contour correctly, but still recognized the contour as being closed;
        % this will only be done once the contour was found to be correctly
        % recognized once, i.e. once trackingVariables.firstdetection equals 1
        if trackingVariables.firstdetection == 1
            if norm( newcenter-trackingVariables.center, 2 ) <= trackingParameters.maxcenterchange
                setNewCenter = 1;
            end
        end
end