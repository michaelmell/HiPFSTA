function void = writeTrackingInformationToLog(contourNrIndex,trackingVariables,programParameters)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% outputTrackingInformation   Function gives output of information on
%%% tracking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fid = fopen([programParameters.data_analysis_directory_path,'/trackingLog.txt'],'A');
% disp(['contourNrIndex = ',num2str(contourNrIndex)])
fprintf(fid,'%s\n',['contourNrIndex = ',num2str(contourNrIndex)]);
% disp(['Number of tracked contour points = ',num2str(length(trackingVariables.xymempos))])
fprintf(fid,'%s\n',['Number of tracked contour points = ',num2str(length(trackingVariables.xymempos))]);
% disp(['Contour closed = ',num2str(trackingVariables.contourclosed)])
fprintf(fid,'%s\n',['Contour closed = ',num2str(trackingVariables.contourclosed)]);
global breakofpixel
% disp(['Value of breakofpixel: ',num2str(breakofpixel)]);
fprintf(fid,'%s\n',['Value of breakofpixel: ',num2str(breakofpixel)]);
% disp(['Contour Index: ',num2str(contourNrIndex)]);
fprintf(fid,'%s\n',['Contour Index: ',num2str(contourNrIndex)]);
% disp(['Contour center coordinates: ' num2str(trackingVariables.center)])
fprintf(fid,'%s\n',['Contour center coordinates: ' num2str(trackingVariables.center)]);
% disp(['Value of trackingVariables.sizeofvesicle: ' num2str(trackingVariables.sizeofvesicle)])
fprintf(fid,'%s\n',['Value of trackingVariables.sizeofvesicle: ' num2str(trackingVariables.sizeofvesicle)]);
% disp(['Value of trackingVariables.delta: ' num2str(trackingVariables.delta)])
fprintf(fid,'%s\n',['Value of trackingVariables.delta: ' num2str(trackingVariables.delta)]);
% disp(['Value of trackingVariables.contourRadius: ' num2str(trackingVariables.contourRadius)])
fprintf(fid,'%s\n',['Value of trackingVariables.contourRadius: ' num2str(trackingVariables.contourRadius)]);
% disp(['Value of trackingVariables.newstartingpixpos: ' num2str(trackingVariables.startingPixelPosition)])
fprintf(fid,'%s\n',['Value of trackingVariables.newstartingpixpos: ' num2str(trackingVariables.startingPixelPosition)]);
fprintf(fid,'\n');
fprintf(fid,'\n');
fclose('all');
