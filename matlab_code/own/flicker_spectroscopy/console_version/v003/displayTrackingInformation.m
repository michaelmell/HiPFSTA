function void = displayTrackingInformation(contourNrIndex,trackingVariables)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% outputTrackingInformation   Function gives output of information on
%%% tracking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(['contourNrIndex = ',num2str(contourNrIndex)])

disp(['Number of tracked contour points = ',num2str(length(trackingVariables.xymempos))])

disp(['Contour closed = ',num2str(trackingVariables.contourclosed)])

global breakofpixel
disp(['Value of breakofpixel: ',num2str(breakofpixel)]);

disp(['Contour Index: ',num2str(contourNrIndex)]);

disp(['Contour center coordinates: ' num2str(trackingVariables.center)])

disp(['Value of trackingVariables.sizeofvesicle: ' num2str(trackingVariables.sizeofvesicle)])

disp(['Value of trackingVariables.delta: ' num2str(trackingVariables.delta)])

disp(['Value of trackingVariables.contourRadius: ' num2str(trackingVariables.contourRadius)])

disp(['Value of trackingVariables.newstartingpixpos: ' num2str(trackingVariables.startingPixelPosition)])