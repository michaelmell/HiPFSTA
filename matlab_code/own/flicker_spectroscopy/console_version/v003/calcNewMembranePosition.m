function trackingVariables = calcNewMembranePosition(image,index2,trackingParameters,trackingVariables)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Calculation of next membrane point according to the tracking algorithm
%%%
%%% in the following the Slopes and positions of the membrane are calculated
%%% in the coordinate-systems. Later the exact position of the membrane is
%%% determined by calculation of the baryo trackingVariables.center of these four points
%%% weighted by the slopes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% membrane position ybar in y-direction at x_i+1, ie: (x_i+1,ybar)
linfitxvalues = (trackingVariables.pixpos(index2,2)-trackingParameters.linfitparameter:trackingVariables.pixpos(index2,2)+trackingParameters.linfitparameter);
linfitintensities = image(trackingVariables.pixpos(index2,1)+trackingVariables.xdirection,trackingVariables.pixpos(index2,2)-trackingParameters.linfitparameter:...
                          trackingVariables.pixpos(index2,2)+trackingParameters.linfitparameter);

p = linearFit(linfitxvalues,linfitintensities); %calculate linear fit straight
        
trackingVariables.Sy = p(1);

if trackingVariables.Sy ==0
   trackingVariables.Sy = 0.00001; 
end

intercepty = p(2);
meanvaluey = meanOneDimension(image(trackingVariables.pixpos(index2,1)+trackingVariables.xdirection,trackingVariables.pixpos(index2,2)-trackingParameters.meanparameter:trackingVariables.pixpos(index2,2)+trackingParameters.meanparameter));
ybar = (meanvaluey-intercepty)/trackingVariables.Sy;

% for old trackingVariables.method
ybar2 = ybar - trackingVariables.pixpos(index2,2);

% membrane position xbar in x-direction at y_i+1, ie: (xbar,y_i+1)
linfitxvalues = (trackingVariables.pixpos(index2,1)-trackingParameters.linfitparameter:trackingVariables.pixpos(index2,1)+trackingParameters.linfitparameter);
linfitintensities = transpose(image(trackingVariables.pixpos(index2,1)-trackingParameters.linfitparameter:....
                                    trackingVariables.pixpos(index2,1)+trackingParameters.linfitparameter,trackingVariables.pixpos(index2,2)+trackingVariables.ydirection));

p = linearFit(linfitxvalues,linfitintensities); %calculate linear fit straight

trackingVariables.Sx = p(1);

if trackingVariables.Sx ==0
   trackingVariables.Sx = 0.00001; 
end

interceptx = p(2);
meanvaluex = meanOneDimension(image(trackingVariables.pixpos(index2,1)-trackingParameters.meanparameter:trackingVariables.pixpos(index2,1)+trackingParameters.meanparameter,trackingVariables.pixpos(index2,2)+trackingVariables.ydirection));
xbar = (meanvaluex - interceptx)/trackingVariables.Sx;

% for old trackingVariables.method
xbar2 = xbar - trackingVariables.pixpos(index2,1);

% membrane position wbar in w-direction at v_i+1, ie: (v_i+1,wbar)
% set the diagonal to be analysed
xposition_at_v_i1 = trackingVariables.pixpos(index2,1)+trackingVariables.vdirection; % analyse at the x-y-podition in direction of v
yposition_at_v_i1 = trackingVariables.pixpos(index2,2)+trackingVariables.vdirection;

wmeancutout = image(xposition_at_v_i1-trackingParameters.diagMeanParameter:xposition_at_v_i1+trackingParameters.diagMeanParameter,...
                    yposition_at_v_i1-trackingParameters.diagMeanParameter:yposition_at_v_i1+trackingParameters.diagMeanParameter);

wlinfitcutout = image(xposition_at_v_i1-trackingParameters.diagLinFitParameter:xposition_at_v_i1+trackingParameters.diagLinFitParameter,...
                      yposition_at_v_i1-trackingParameters.diagLinFitParameter:yposition_at_v_i1+trackingParameters.diagLinFitParameter);

meanintensities = perpdiag(wmeancutout);
linfitintensities = perpdiag(wlinfitcutout);

% the distance needs to be corrected since the pixels in the diagonals are 1.4142
% pixels apart ???
linfitxvalues = (-trackingParameters.diagLinFitParameter:trackingParameters.diagLinFitParameter); 
p = linearFit(linfitxvalues,linfitintensities); %calculate linear fit straight

trackingVariables.Sw = p(1);

if trackingVariables.Sw ==0
   trackingVariables.Sw = 0.00001;
end

interceptw = p(2);
meanvaluew = meanOneDimension(meanintensities);
wbar = (meanvaluew - interceptw)/trackingVariables.Sw;

% membrane position vbar in v-direction at w_i+1, ie: (vbar,w_i+1)
% set the diagonal to be analysed
xposition_at_w_i1 = trackingVariables.pixpos(index2,1) - trackingVariables.wdirection; % analyse at the x-y-position in direction of w
yposition_at_w_i1 = trackingVariables.pixpos(index2,2) + trackingVariables.wdirection;

vmeancutout = image(xposition_at_w_i1-trackingParameters.diagMeanParameter:xposition_at_w_i1+trackingParameters.diagMeanParameter,...
                    yposition_at_w_i1-trackingParameters.diagMeanParameter:yposition_at_w_i1+trackingParameters.diagMeanParameter);

vlinfitcutout = image(xposition_at_w_i1-trackingParameters.diagLinFitParameter:xposition_at_w_i1+trackingParameters.diagLinFitParameter,...
                      yposition_at_w_i1-trackingParameters.diagLinFitParameter:yposition_at_w_i1+trackingParameters.diagLinFitParameter);

meanintensities = transpose(diag(vmeancutout));
linfitintensities = transpose(diag(vlinfitcutout));

% the distance needs to be corrected since the pixels in the diagonals are 1.4142
% pixels apart ???
linfitxvalues = (-trackingParameters.diagLinFitParameter:trackingParameters.diagLinFitParameter);
p = linearFit(linfitxvalues,linfitintensities); %calculate linear fit straight

trackingVariables.Sv = p(1);

if trackingVariables.Sv ==0
   trackingVariables.Sv = 0.00001; 
end

interceptv = p(2);
meanvaluev = meanOneDimension(meanintensities);

vbar = (meanvaluev - interceptv)/trackingVariables.Sv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% calculation of baryocenter of the calculated points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% the coordinates are calculated using the absolute values of the
% slopes
trackingVariables.Sx = abs(trackingVariables.Sx);
trackingVariables.Sy = abs(trackingVariables.Sy);
trackingVariables.Sv = abs(trackingVariables.Sv);
trackingVariables.Sw = abs(trackingVariables.Sw);

% calculate xtild and ytild
xtild = (trackingVariables.Sy*trackingVariables.xdirection + trackingVariables.Sx*xbar2 + trackingVariables.Sw*(trackingVariables.vdirection*0.707106781 - (wbar/1.414213562)) + trackingVariables.Sv*((-trackingVariables.wdirection)*0.707106781 + (vbar/1.414213562)))/(trackingVariables.Sx+trackingVariables.Sy+trackingVariables.Sv+trackingVariables.Sw);
ytild = (trackingVariables.Sy*ybar2 + trackingVariables.Sx*trackingVariables.ydirection + trackingVariables.Sw*(trackingVariables.vdirection*0.707106781 + (wbar/1.414213562)) + trackingVariables.Sv*(trackingVariables.wdirection*0.707106781 + (vbar/1.414213562)))/(trackingVariables.Sx+trackingVariables.Sy+trackingVariables.Sv+trackingVariables.Sw);

% calculate membrane position from YTILD and XTILD
trackingVariables.xymempos(index2+1,1) = trackingVariables.pixpos(index2,1) + xtild;
trackingVariables.xymempos(index2+1,2) = trackingVariables.pixpos(index2,2) + ytild;

% coordinate transformation to the vw-coordinate system
trackingVariables.vwmempos(index2+1,1) = trackingVariables.xymempos(index2,1) * 0.70710678 + trackingVariables.xymempos(index2,2) * 0.70710678; % trafo to v-axis
trackingVariables.vwmempos(index2+1,2) = trackingVariables.xymempos(index2,2) * 0.70710678 - trackingVariables.xymempos(index2,1) * 0.70710678; % trafo to w-axis