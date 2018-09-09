function void = console_tracking(parameterStruct)

function bkgrData = calculateBkgrImage(parameterStruct)
    disp('Calculating background image ...');
    background_image_data_directory_path = parameterStruct.background_image_data_directory_path;
    fileList = dir([background_image_data_directory_path,'/*.',parameterStruct.filePostfix]);
    tmp = double( imread([background_image_data_directory_path,'/',fileList(1).name]) );
    bkgrData = zeros(size(tmp));

    for ind = 1:length(fileList)
        bkgrData = bkgrData + double( imread([background_image_data_directory_path,'/',fileList(ind).name]) );
    end
    bkgrData = bkgrData/length(fileList);
end

%%
%% Return: true if the environment is Octave.
%%
function retval = isOctave
  persistent cacheval;  % speeds up repeated calls

  if isempty (cacheval)
    cacheval = (exist ("OCTAVE_VERSION", "builtin") > 0);
  end

  retval = cacheval;
end

if(isOctave())
  pkg load statistics;
end

% % get new tracking settings settings
% hTrackingMenuGUI = gcbf;
% hTrackingMenuData = guidata(hTrackingMenuGUI);
% handles = hTrackingMenuData;

% parameterStruct = handles.parameterStruct;
% parameterStruct = handles.parameterStruct;
[programParameters,trackingParameters,trackingVariables] = initializeTracking(parameterStruct);
parameterStruct = createImageFileList(parameterStruct);

if ~exist(programParameters.data_analysis_directory_path,'dir')
    mkdir(programParameters.data_analysis_directory_path);
end
save([programParameters.data_analysis_directory_path,'/parameterStruct','.mat'],'parameterStruct','-mat');
save([programParameters.data_analysis_directory_path,'/programParameters','.mat'],'programParameters','-mat');
save([programParameters.data_analysis_directory_path,'/trackingParameters','.mat'],'trackingParameters','-mat');

contourNrIndex = 1;

% parameterStruct = loadImages(parameterStruct);
% load image
% [parameterStruct.image,parameterStruct.displayimage] = ...
%     loadImage(parameterStruct.image_data_directory_path,parameterStruct.image_file_list{parameterStruct.indexOfSelectedImage});

% load background and do background correction
if ~isempty(parameterStruct.background_file)
    parameterStruct.background = double( imread(parameterStruct.background_file) ); % creates nxn-matrix of grayscale-image with intensity values
end

if ~isempty(parameterStruct.background_image_data_directory_path)
%     parameterStruct.background = double( imread(parameterStruct.background_file) ); % creates nxn-matrix of grayscale-image with intensity values
    parameterStruct.background = calculateBkgrImage(parameterStruct);
end

firstdetection_tmp = 0;
trackingVariables.contourclosed = 0;
trackingVariables.firstdetection = firstdetection_tmp;

fileImageIndexes = 1:length(parameterStruct.image_file_list);
fileImageIndexes(parameterStruct.ignoredImageIndexes) = [];

fileImageIndex = fileImageIndexes(1);

if parameterStruct.runInteractive == 1
    figure(1);
    disp(['Current image file: ',parameterStruct.image_file_list{fileImageIndex}]);
    parameterStruct.image = double( imread([parameterStruct.image_data_directory_path,'/',parameterStruct.image_file_list{fileImageIndex}])); % creates nxn-matrix of grayscale-image with intensity values
    imageData = parameterStruct.image;
    imageData(imageData==Inf) = 0;
    imagesc(imageData,[0,mean(mean(imageData))]);
    daspect([1 1 1]);
    disp('Check find the starting point of the contour and modifiy the parameter ''startingPixelPosition'' in the config-file.');
    pause
    close(figure(1));
end

if ~exist(programParameters.data_analysis_directory_path,'dir')
        disp(char(10));
        disp(['The directory for saving the tracking output does not exist. Please create it first:',char(10), programParameters.data_analysis_directory_path]);
else
    if parameterStruct.runInteractive == 1
        tmp = dir([programParameters.data_analysis_directory_path,'/*.*']);
        if length(tmp)>2
            ButtonName = questdlg('Warning: Directory is not empty. Files might be overwritten. Continue?', ...
                         'Overwrite Files', ...
                         'Yes', 'No', 'Yes');
            switch ButtonName
                case 'No'
                    disp('Tracking aborted.');
                    return;
                case 'Yes'
            end
        end
    end
end

% while parameterStruct.continuousTrackingOn == 1
%     for fileImageIndex = parameterStruct.selected_image_files

contourCenters = nan(length(fileImageIndexes),2);

for fileImageIndex = fileImageIndexes
    disp(['Current image file: ',parameterStruct.image_file_list{fileImageIndex}]);
    parameterStruct.image = double( imread([parameterStruct.image_data_directory_path,'/',parameterStruct.image_file_list{fileImageIndex}])); % creates nxn-matrix of grayscale-image with intensity values

    % load image
    correctedImage = parameterStruct.image;
    % do background correction if requested

    if isfield(parameterStruct,'background')
        background = parameterStruct.background;
        correctedImage = correctedImage./background;
    end
    
    % do filtering if requested
    if ~isempty(parameterStruct.gaussian_filter_width)
        correctedImage = filterWithGaussian(correctedImage, parameterStruct.gaussian_filter_width);
    end
    parameterStruct.correctedImage = correctedImage;
    
    if ~isempty(parameterStruct.imageContrastEnhancementFactor)
        imageData = parameterStruct.imageContrastEnhancementFactor * correctedImage;
    else
        imageData = correctedImage;
    end
%         parameterStruct = calculateCorrectedImage(parameterStruct);
%         imageData = parameterStruct.correctedImage;

%         parameterStruct.image = imageData;
%         parameterStruct = calculateCorrectedImage(parameterStruct);
%         imageData = parameterStruct.correctedImage;

    %determine the size of the image to later determine the middle line of the
    %image on which to find the membrane (see below)
    [trackingVariables.imagexmax trackingVariables.imageymax] = size(imageData); % max coordinates in the x- and y-directions

    % set starting point for the tracking of the following image,
    % depending on whether the contour was detected correctly or not
% keyboard

    if trackingVariables.firstdetection == 0
%             trackingVariables.startingPixelPosition = findFirstStartingPixelPosition(imageData,trackingVariables.center,trackingParameters,trackingVariables);
% keyboard
        trackingVariables.startingPixelPosition = trackingParameters.startingPixelPosition;
    else
        trackingVariables.startingPixelPosition = startingPixelPosition_tmp;
    end

    % reset control variables start tracking
    trackingVariables.contourclosedindex(contourNrIndex) = 0;
try

    % track image to extract contour
    [trackingVariables.xymempos,trackingVariables.pixpos,trackingVariables.contourclosed] = contourTrack(imageData,trackingVariables.startingPixelPosition,trackingParameters,trackingVariables);

    % show zoomed image of selected vesicle
%     trackingVariables.displayimage = markContourCoordinates(trackingVariables.displayimage,trackingVariables);
%     trackingVariables.VesicleROI = getVesicleROI(trackingVariables.displayimage,trackingVariables.center,trackingVariables.sizeofvesicle,trackingParameters.surroundings);
%     programParameters.hFig = showImage(trackingVariables.VesicleROI,trackingParameters.colorMapType);

    % mark contour in the displayed image
%         draw_image_to_axes(handles.tracked_vesicle_data_axes,imageData);
%         imageData(imageData==Inf) = 0;
%         imageData(isnan(imageData)) = 0;
%         imagesc(imageData);
    subplot(1,2,1);
    imageData(imageData==Inf) = 0;
    imagesc(imageData,[0,nanmean(nanmean(imageData))]);
    daspect([1 1 1]);
    % draw tracked contour
    drawOverlay(gca,trackingVariables.pixpos,'w',2);
    drawOverlay(gca,trackingVariables.xymempos,'k',2);
%         drawOverlay(handles.tracked_vesicle_data_axes,trackingVariables.pixpos,'b',2);
%         drawOverlay(handles.tracked_vesicle_data_axes,trackingVariables.xymempos,'r',2);
    % draw starting point of track and the vesicle center 
%         drawOverlay(handles.tracked_vesicle_data_axes,trackingVariables.startingPixelPosition,'w',8);
%         drawOverlay(handles.tracked_vesicle_data_axes,trackingVariables.center,'r',8);
    drawOverlay(gca,trackingVariables.startingPixelPosition,'w',8);
%         drawOverlay(gca,trackingVariables.center,'r',8);
    if trackingVariables.firstdetection
        subplot(1,2,2);
        imageData(imageData==Inf) = 0;
        imagesc(imageData,[0,nanmean(nanmean(imageData))]);
        daspect([1 1 1]);
        % draw tracked contour
        drawOverlay(gca,trackingVariables.pixpos,'w',2);
        drawOverlay(gca,trackingVariables.xymempos,'k',2);
        drawOverlay(gca,trackingVariables.startingPixelPosition,'w',8);
        [xmin,xmax,ymin,ymax] = getVesicleExtension(trackingVariables.xymempos);
        margin = 5;
        xlim([xmin-margin,xmax+margin]);
        ylim([ymin-margin,ymax+margin]);
    end
    drawnow;
    pause(0.05);
    
    if trackingVariables.contourclosed == 1
        if trackingVariables.firstdetection == 0
            if parameterStruct.runInteractive == 1
                trackingVariables = userRequestTrackingSuccess(trackingVariables);
            else
                trackingVariables.firstdetection = 1;
            end
%                 firstdetection_tmp = trackingVariables.firstdetection;
            [newcenter,contourCircumference,contourRadius] = calcContourProperties(trackingVariables.xymempos);
            trackingVariables.center = newcenter;
        end
        %%% find starting point for membrane recognition
        if trackingVariables.firstdetection == 1
            trackingVariables.sizeofvesicle = sqrt((trackingVariables.startingPixelPosition(1)-trackingVariables.center(1))^2 + ...
                                                   (trackingVariables.startingPixelPosition(2)-trackingVariables.center(2))^2);
            trackingVariables.delta = trackingVariables.startingPixelPosition - trackingVariables.center;
            trackingVariables.startingPixelPosition = setNewTrackingStartingPoint(contourNrIndex,trackingVariables.startingPixelPosition,trackingVariables.center,trackingVariables.delta,trackingVariables.xymempos);
            startingPixelPosition_tmp = trackingVariables.startingPixelPosition;
        end
    end

    % write tracked coordinates to structure
    trackingVariables.contour.coordinates{contourNrIndex} = trackingVariables.xymempos;
    trackingVariables.contour.pixelPositions{contourNrIndex} = trackingVariables.pixpos;
    
    % test the tracked contour
    trackingVariables = testContour(contourNrIndex,trackingVariables,trackingParameters);

    % output control information
    if trackingVariables.firstdetection == 1
        displayTrackingInformation(contourNrIndex,trackingVariables);
        writeTrackingInformationToLog(contourNrIndex,trackingVariables,programParameters);
    end
    
catch exception
    display('###################### An error occured! ######################')
    % display(getReport(exception)); 
    display(exception); 
end


contourCoordinates = trackingVariables.contour.coordinates;
contourCenters(contourNrIndex,:) = trackingVariables.center;

if mod(fileImageIndex,parameterStruct.saveInterval) == 0
    disp('Saving intermediate tracked data ...');
    
    save([programParameters.data_analysis_directory_path,'/contourCoordinates','.mat'],'contourCoordinates');
    save([programParameters.data_analysis_directory_path,'/contourCenters','.mat'],'contourCenters');
    saveTrackingResults(contourNrIndex,trackingVariables,programParameters)
end

contourNrIndex = contourNrIndex + 1;

%     % stop tracking if terminated by user input
%     if parameterStruct.continuousTrackingOn == 0
%         break;
%     end
end
% end

% % write settings to main GUI and close submenu
% handles.trackingVariables = trackingVariables;
% hTrackingMenuData = handles;
% guidata(hTrackingMenuGUI,hTrackingMenuData);

%%% save tracked contours and information on tracking
contourNrIndex = contourNrIndex - 1;
saveTrackingResults(contourNrIndex,trackingVariables,programParameters)

% this diplays a message box with information regarding the tracking
% process
if parameterStruct.runInteractive == 1
    msgbox(sprintf('Tracked Contours: %i \nClosed Contours: %i \nNot Closed Contours: %i', ...
        contourNrIndex, ...
        length(find(trackingVariables.contourclosedindex(1:contourNrIndex) == 1)), ...
        length(find(trackingVariables.contourclosedindex(1:contourNrIndex) == 0))),'Results','none','modal')
end

end