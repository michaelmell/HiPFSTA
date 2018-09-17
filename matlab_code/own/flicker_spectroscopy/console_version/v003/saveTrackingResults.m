function void = saveTrackingResults(contourNrIndex,trackingVariables,programParameters)

% create path were to save the results
% pathToResultsDirectory = [programParameters.data_analysis_directory_path,'/'];
% pathToResultsDirectory = [programParameters.data_analysis_directory_path,'/',programParameters.data_analysis_configFileName];

% create directory for saving contours
% mkdir([pathToResultsDirectory,'/','contours',]);

%%% save tracked contours
contourCoordinates = trackingVariables.contour.coordinates;
save([programParameters.data_analysis_directory_path,'/contourCoordinates','.mat'],'contourCoordinates','-mat');
% for index = 1:contourNrIndex
%     xymempos = trackingVariables.contour.coordinates{index};
%     save([pathToResultsDirectory,'/contours/contour',num2str(index),'.txt'],'xymempos','-ascii'); % save the contour
% end

%%% save pixel-positions tracked contours
pixelPositions = trackingVariables.contour.pixelPositions;
save([programParameters.data_analysis_directory_path,'/pixelPositions','.mat'],'pixelPositions','-mat');

%%% save tracking output
trackingVariables = rmfield(trackingVariables,'contour');
save([programParameters.data_analysis_directory_path,'/trackingVariables','.mat'],'trackingVariables','-mat');

% make directory that will be holding the information on the specific
% project
% mkdir([pathToResultsDirectory,'/results']);

% save trackingVariables.contourclosedindex
tmp = trackingVariables.contourclosedindex;
save([programParameters.data_analysis_directory_path,'/contourclosedindex.txt'],'tmp','-ascii'); % save the contour

% this save the data displayed in the message box below into a log-file for
% later reference
fid = fopen([programParameters.data_analysis_directory_path,'/nrOfClosedContours.txt'],'w');
fprintf(fid,'Tracked Contours: %i \nClosed Contours: %i \nNot Closed Contours: %i', ...
    contourNrIndex, ...
    length(find(trackingVariables.contourclosedindex(1:contourNrIndex) == 1)), ...
    length(find(trackingVariables.contourclosedindex(1:contourNrIndex) == 0)));
fclose('all');
