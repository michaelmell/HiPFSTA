function void = console_tracking(handles)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% tracking    this function is used for tracking the contour of the
%%% vesicles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

parameterStruct = handles.parameterStruct;
[programParameters,trackingParameters,trackingVariables] = initializeTracking(parameterStruct);

% setup up file system information
programParameters = getNumberOfImages(programParameters);

% trackingVariables.contourRadius = handles.trackingVariables.contourRadius;
% trackingVariables.startingPixelPosition = handles.trackingVariables.startingPixelPosition;
trackingVariables = handles.trackingVariables;

contourNrIndex = 1;

% % display waitbar
% trackingVariables.waitbarHandle = awaitbar(0,'Tracking Contours, please wait...');
% abort = false; % variable for waitbar

%%% loop through images and track vesicle contour
for fileImageIndex = parameterStruct.selected_image_files
% for contourNrIndex = 1:programParameters.contourNr % loop to go through the contours
    % find the contours
%     trackingVariables = findContour(contourNrIndex,programParameters,trackingParameters,trackingVariables);
try
    % load image
%     [imageData,trackingVariables.displayimage] = loadImage(programParameters.image_data_directory_path,parameterStruct.image_file_list{fileImageIndex});
%     parameterStruct.image = imageData;
    parameterStruct = loadImages(parameterStruct);
    parameterStruct = calculateCorrectedImage(parameterStruct);
    imageData = parameterStruct.correctedImage;

    % set starting point for the tracking of the following image,
    % depending on whether the contour was detected correctly or not
    trackingVariables.startingPixelPosition = findFirstStartingPixelPosition(imageData,trackingVariables.center,trackingParameters,trackingVariables);

    % reset control variables start tracking
    trackingVariables.contourclosedindex(contourNrIndex) = 0;

    % track image to extract contour
    [trackingVariables.xymempos,trackingVariables.pixpos,trackingVariables.contourclosed] = contourTrack(imageData,trackingVariables.startingPixelPosition,trackingParameters,trackingVariables);

    % write tracked coordinates to structure
    trackingVariables.contour.coordinates{contourNrIndex} = trackingVariables.xymempos;

    % test the tracked contour
    trackingVariables = testContour(contourNrIndex,trackingVariables,trackingParameters);

%     % show zoomed image of selected vesicle
%     trackingVariables.displayimage = markContourCoordinates(trackingVariables.displayimage,trackingVariables);
%     trackingVariables.VesicleROI = getVesicleROI(trackingVariables.displayimage,trackingVariables.center,trackingVariables.sizeofvesicle,trackingParameters.surroundings);
%     programParameters.hFig = showImage(trackingVariables.VesicleROI,trackingParameters.colorMapType);

    % mark contour in the displayed image
    draw_image_to_axes(handles.tracked_vesicle_data_axes,imageData);
    % draw tracked contour
    drawOverlay(handles.tracked_vesicle_data_axes,trackingVariables.pixpos,'b',2);
    drawOverlay(handles.tracked_vesicle_data_axes,trackingVariables.xymempos,'r',2);
    % draw starting point of track and the vesicle center
    drawOverlay(handles.tracked_vesicle_data_axes,trackingVariables.startingPixelPosition,'w',8)
    drawOverlay(handles.tracked_vesicle_data_axes,trackingVariables.center,'r',8)
    drawnow;
    pause(0.01);

    % set starting pixel for tracking of the following image
    if trackingVariables.contourclosed == 1
        % set new starting point for tracking the following image
        trackingVariables.startingPixelPosition = setNewTrackingStartingPoint(contourNrIndex,trackingVariables.startingPixelPosition,trackingVariables.center,trackingVariables.delta,trackingVariables.xymempos);
    end

    % output control information
    outputTrackingInformation(contourNrIndex,trackingVariables);
keyboard
catch exception
    display('###################### An error occured! ######################')
    display(getReport(exception));
    keyboard
end
%     %%%  update waitbar and check if process was aborted
%     trackingVariables.waitbarHandle2 = awaitbar(contourNrIndex/length(parameterStruct.selected_image_files),trackingVariables.waitbarHandle,'Running the process','Progress');
%     if abort % check if user wants to abort
%         close(trackingVariables.waitbarHandle2);
%         break; % abort the process by clicking abort button
%     elseif isempty(trackingVariables.waitbarHandle2)
%         break; % break the process when closing the figure
%     end
    
    contourNrIndex = contourNrIndex + 1;
end
contourNrIndex = contourNrIndex - 1;

%%% save tracked contours and information on tracking
saveTrackingResults(contourNrIndex,trackingVariables,programParameters)

% this diplays a message box with information regarding the tracking
% process
msgbox(sprintf('Tracked Contours: %i \nClosed Contours: %i \nNot Closed Contours: %i', ...
    contourNrIndex, ...
    length(find(trackingVariables.contourclosedindex(1:contourNrIndex) == 1)), ...
    length(find(trackingVariables.contourclosedindex(1:contourNrIndex) == 0))),'Results','none','modal')