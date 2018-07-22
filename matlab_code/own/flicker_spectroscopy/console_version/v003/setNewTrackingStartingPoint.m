function newTrackingStartingPoint = setNewTrackingStartingPoint(contourNrIndex,oldStartingPoint,center,delta,xymempos)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% newTrackingStartingPoint Function to set the new starting point for
%%% the tracking of the following contour, once the first contour has been
%%% tracked correctly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pointAboveCenter = (xymempos(:,1) < center(1));
xymemposTMP = xymempos(pointAboveCenter,:); % only use point above the center to find new starting point

% newTrackingStartingPoint(1) = round(center(1));
tmp = find(round(xymemposTMP(:,2)) == round(center(2)));

if isempty(tmp) == 0
    if length(tmp)> 1
        tmp = tmp(1);
    end
end


if isempty(tmp) == 1
    tmp = find(round(xymemposTMP(:,2)) == round(center(2) - 1));
elseif isempty(tmp) == 1
    tmp = find(round(xymemposTMP(:,2)) == round(center(2) + 1));
elseif isempty(tmp) == 1
    tmp = find(round(xymemposTMP(:,2)) == round(center(2) - 2));
elseif isempty(tmp) == 1
    tmp = find(round(xymemposTMP(:,2)) == round(center(2) + 2));
end

if length(tmp)> 1
    tmp = tmp(1);
end

newTrackingStartingPoint = round(xymemposTMP(tmp,:));

if isempty(newTrackingStartingPoint)
    newTrackingStartingPoint = oldStartingPoint;
else
    distanceBetweenOldAndNewStartingsPoint = sqrt((oldStartingPoint(1)-newTrackingStartingPoint(1))^2 + ...
                                                  (oldStartingPoint(2)-newTrackingStartingPoint(2))^2);
    if distanceBetweenOldAndNewStartingsPoint > 10 % control that the new starting point does not jump too much
        newTrackingStartingPoint = oldStartingPoint;
    end
end


% original
% newTrackingStartingPoint = round(center + delta);

% % newTrackingStartingPoint(1) = round(center(1));
% tmp = find(round(xymempos(:,2)) == round(center(2)));
% if isempty(tmp(1)) == 1
%     tmp = find(round(xymempos(:,2)) == round(center(2) - 1));
% elseif isempty(tmp(1)) == 1
%     tmp = find(round(xymempos(:,2)) == round(center(2) + 1));
% end

% get smallest y-value for the starting coordinate, so that we start at
% the top of the vesicle

% indexOfMinimumYValue = (round(xymempos(tmp(:),1)) == min(round(xymempos(tmp(:),1))));
% newTrackingStartingPoint = round(xymempos(tmp(indexOfMinimumYValue),:));
% 
% keyboard

% newTrackingStartingPoint = round(xymempos(tmp(2),:));
% newTrackingStartingPoint = round(xymempos(tmp(1),:));

% TODO: remove this!!!
% newTrackingStartingPoint = [276   526];
