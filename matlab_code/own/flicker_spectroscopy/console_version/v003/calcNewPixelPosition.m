function trackingVariables = calcNewPixelPosition(image,index2,trackingParameters,trackingVariables)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Determination of new position on pixelgrid 
%%% (i.e.: determination of trackingVariables.pixpos(index2+1,:))
%%%
%%% For this there are 4 methodes to be used to find a new pixel-position 
%%% trackingVariables.pixpos(index2+1,:). These are tested successively until one is 
%%% successful.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Methode 1:
% round the coordinates of the membrane position to the next integer. If the
% resulting pixel-position trackingVariables.pixpos(index2+1,:) equals trackingVariables.pixpos(index2,:), the next
% condition is tested
trackingVariables.pixpos(index2+1,:) = round(trackingVariables.xymempos(index2+1,:));

% If the determination of the new pixelposition by rounding of the membran position fails try next determination trackingVariables.method 
if pixelpositiontest(trackingVariables.pixpos,trackingParameters.pixeltestparameter,index2)==1 % this checks if trackingVariables.method 1 was used
    trackingVariables.method(index2) = 1;
    trackingVariables.methodinternaldirection(index2) = 0;
end


% check the other methods by hierarchy
if pixelpositiontest(trackingVariables.pixpos,trackingParameters.pixeltestparameter,index2)==0 % use trackingVariables.method 2 to find next pixel position
    [trackingVariables.pixpos(index2+1,:),trackingVariables.methodinternaldirection(index2)] = newpixposmethod2(trackingVariables.pixpos,index2,trackingVariables.Sx,trackingVariables.Sy,trackingVariables.Sv,trackingVariables.Sw,trackingVariables.xdirection,trackingVariables.ydirection,trackingParameters.slopediff);
    trackingVariables.method(index2) = 2;
    if pixelpositiontest(trackingVariables.pixpos,trackingParameters.pixeltestparameter,index2)==0 % use trackingVariables.method 3 to find next pixel position
        [trackingVariables.pixpos(index2+1,:),trackingVariables.methodinternaldirection(index2)] = newpixposmethod3(trackingVariables.xymempos,trackingVariables.pixpos,index2);
        trackingVariables.method(index2) = 3;
        if pixelpositiontest(trackingVariables.pixpos,trackingParameters.pixeltestparameter,index2)==0 % use trackingVariables.method 4 to find next pixel position
            [trackingVariables.pixpos(index2+1,:),trackingVariables.methodinternaldirection(index2)] = newpixposmethod4(trackingVariables.xymempos,trackingVariables.pixpos,index2,trackingParameters.methode4parameter);
            trackingVariables.method(index2) = 4;
            if pixelpositiontest(trackingVariables.pixpos,trackingParameters.pixeltestparameter,index2)==0 % use methode 5 ("e.g.: the old trackingVariables.method") to find next pixel position;
                [trackingVariables.pixpos(index2+1,:),trackingVariables.methodinternaldirection(index2),returnblank] = newpixposmethodold(trackingVariables.pixpos(index2,:),image,trackingVariables.xdirection,trackingVariables.ydirection,trackingParameters.meanparameter,trackingParameters.linfitparameter);
                trackingVariables.method(index2) = 5;
            end
        end
    end
end