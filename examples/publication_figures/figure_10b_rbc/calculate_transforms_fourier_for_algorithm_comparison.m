%%
run('../../../matlab_code/setPaths.m');  

%% create RBC dataset
close all;
clear all;
clear classes;

basePath = '../../';
% basePath = '/media/data_volume/figure_10_files/rbc/healthy/2014-05-05/';

datasetPath = {'rbc/tracking/matlab_tracking_002/', ...
               };
datasetLabelAlternative = {'rbc_healthy_2014-05-05_rbc_4_movie_1_tracking_matlab_tracking_002', ...
                          };

% indices = (1:length(datasetLabelAlternative));
% indices = (5:length(datasetLabelAlternative));
indices = 1;
% indices = (4:7);
% indices = (2:3);

% savePath = datasetPath;
savePath = {[basePath,'/',datasetPath{1}], ...
           };
% indices = 5;

fourierSeriesFilenameEachCenter = 'fourierSeriesCenterOfEachContourCorrected.mat';
fourierSeriesFilenameMeanCenter = 'fourierSeriesMeanCenterCorrected.mat';

for index = indices
    idString = datasetPath{index};
    name = datasetLabelAlternative{index};
    path = [basePath,datasetPath{index},'/'];

    % set parameters
    dataset = flickeringDataClass();
    dataset.setIdString(name);
    dataset.setLabelString(name);
    dataset.loadContours( path );
    dataset.setNrOfModes(1024);
    dataset.setResolution(50.0e-09);
    dataset.testContours();
    dataset.calculateCircumference();
    dataset.calculateBarioCenter();

    % create version with center for each profile
    if( ~exist([savePath{index},'/',fourierSeriesFilenameEachCenter],'file'))
        dataset.setReferenceCenterMethod('forEachContour');
        dataset.calculatePolarCoordinates();
    %     dataset.calculateFourierTransform();
        dataset.calculateFourierTransformNEW2();
%         dataset.calculateFourierTransformNEW();
        fourierSeries = dataset.fourierseries;
        save([savePath{index},'/',fourierSeriesFilenameEachCenter],'fourierSeries');
    else
       disp(['File exists: ',[savePath{index},'/',fourierSeriesFilenameEachCenter]]) 
    end
    
    if( ~exist([savePath{index},'/',fourierSeriesFilenameMeanCenter],'file'))
        dataset.setReferenceCenterMethod('meanCenter');
        dataset.calculatePolarCoordinates();
    %     dataset.calculateFourierTransform();
        dataset.calculateFourierTransformNEW2();
%         dataset.calculateFourierTransformNEW();
        fourierSeries = dataset.fourierseries;
        save([savePath{index},'/',fourierSeriesFilenameMeanCenter],'fourierSeries');
    else
       disp(['File exists: ',[savePath{index},'/',fourierSeriesFilenameMeanCenter]]) 
    end

    radiusSeries = dataset(1).getRadiusSeries;
    save([savePath{index},'/radiusSeries'],'radiusSeries');
    circumferenceSeries = dataset(1).getCircumferenceSeries;
    save([savePath{index},'/circumferenceSeries'],'circumferenceSeries');
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
close all;
clear all;
clear classes;

fourierSeriesFilenameEachCenter = 'fourierSeriesCenterOfEachContourCorrected.mat';
fourierSeriesFilenameMeanCenter = 'fourierSeriesMeanCenterCorrected.mat';

basePath = {...
            '../../'; ...
            };

basePaths = {...
             basePath{1}, ...
             };
        
datasetPath = { ...
               [basePath{1},'/','rbc/tracking/tracking_000/'], ...
               };

% check that paths exist before starting
for datasetIndex = 1:length(datasetPath)
    if ~exist(datasetPath{datasetIndex},'dir')
        error(['Dataset does not exist: ',char(10),datasetPath{datasetIndex}])
    end
end
           
savePath = datasetPath;

% mainName = 'rbc_';
mainName = '';
datasetLabel = {};
for index = 1:length(datasetPath)
    suffix = strrep(datasetPath{index},[basePaths{index},'/'],''); % remove basePath
    suffix = strrep(suffix,'/','_');
    datasetLabel{index} = [mainName,suffix];
end

% idString = datasetLabel{index};
% name = datasetLabelAlternative{index};

% indexes = 1:length(datasetLabel);
% indexes = 3:length(datasetLabel);
indexes = [1];

% pixelSizes = [50.0e-09,50.0e-09,50.0e-09,50.0e-09];

for datasetIndex = indexes
    % set parameters
    dataset = flickeringDataClass();
    dataset.setIdString(datasetLabel{datasetIndex});
    dataset.setLabelString(datasetLabel{datasetIndex});
    dataset.setResolution(50.0e-09);
    % dataset.loadPythonTrackingSettings_v000( datasetPath );
    % dataset.setNrOfModes(50);
    % dataset.loadContours( datasetPath );
    dataset.setNrOfModes(1024);
    
    dataset.loadPythonTrackingContours_v000( datasetPath{datasetIndex} );
%     dataset.testContours();
    dataset.calculateCircumference();
    dataset.calculateBarioCenter();

    % create version with center for each profile
%     dataset.setReferenceCenterMethod('forEachContour');
%     dataset.calculatePolarCoordinates();
%     dataset.calculateFourierTransformNEW();
%     save([savePath{datasetIndex},'/',datasetLabel{datasetIndex},'_center_of_each_mode.mat'],'dataset');

    %     dataset.loadImageInterpolationSettings_v002( path );
    %     dataset.setResolution(50.0e-09);
    %     dataset.setNrOfModes(50);
    %     dataset.calculateCartesianCoordinates();

    %
%     % create version with averaged center of profiles
%     dataset.setReferenceCenterMethod('meanCenter');
%     dataset.calculatePolarCoordinates();
%     dataset.calculateFourierTransformNEW2();
%     save([savePath{datasetIndex},'/',datasetLabel{datasetIndex},'_averaged_center.mat'],'dataset');

        % create version with center for each profile
    fileName = [savePath{datasetIndex},'/',fourierSeriesFilenameEachCenter];
    if( ~exist(fileName,'file'))
        dataset.setReferenceCenterMethod('forEachContour');
        dataset.calculatePolarCoordinates();
    %     dataset.calculateFourierTransform();
        dataset.calculateFourierTransformNEW2();
%         dataset.calculateFourierTransformNEW();
        fourierSeries = dataset.fourierseries;
        save(fileName,'fourierSeries');
    else
       disp(['File exists: ',fileName]) 
    end
    
    fileName = [savePath{datasetIndex},'/',fourierSeriesFilenameMeanCenter];
    if( ~exist(fileName,'file'))
        dataset.setReferenceCenterMethod('meanCenter');
        dataset.calculatePolarCoordinates();
    %     dataset.calculateFourierTransform();
        dataset.calculateFourierTransformNEW2();
%         dataset.calculateFourierTransformNEW();
        fourierSeries = dataset.fourierseries;
        save(fileName,'fourierSeries');
        radiusSeries = dataset(1).getRadiusSeries;
        save([savePath{index},'/radiusSeries'],'radiusSeries');
        circumferenceSeries = dataset(1).getCircumferenceSeries;
        save([savePath{index},'/circumferenceSeries'],'circumferenceSeries');
    else
       disp(['File exists: ',fileName]) 
    end
end

