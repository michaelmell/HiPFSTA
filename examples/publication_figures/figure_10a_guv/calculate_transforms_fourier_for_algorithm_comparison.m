%%
% run('/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/setPaths.m');
run('C:\Private\PhD_Publications\Publication_of_Algorithm\2018-06-13__figure_10_reproduction_code\matlab_functions/setPaths.m');  

%% create RBC dataset
close all;
clear all;
clear classes;

basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/examples/';
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

return
%% create POPC dataset
close all;
clear all;
clear classes;

% /media/data_volume/figure_10_files/guv/popc/2014-06-03/vesicle_4/movie_1_tracking

% basePath = '/media/lisa_3/figure_10_analysis_mike_2016-05-05/guv_analysis/popc/2014-06-03/';
basePath = 'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/examples/';
% basePath = '/media/data_volume/figure_10_files/guv/popc/2014-06-03/';

datasetPath = {'guv_popc/tracking/matlab_tracking_002/', ...
%                'vesicle_4/movie_1_tracking/tracking_001/', ...
               };
datasetLabelAlternative = {'guv_popc_2014-06-03_vesicle_4_movie_1_tracking_matlab_tracking_002', ...
%                            'guv_popc_2014-06-03_vesicle_4_movie_1_tracking_tracking_001', ...
                          };

% indices = (1:length(datasetLabelAlternative));
% indices = (5:length(datasetLabelAlternative));
indices = 1;
% indices = (4:7);
% indices = (2:3);

savePath = {[basePath,datasetPath{1}], ...
%             [basePath,'/vesicle_4/movie_1_tracking/tracking_001/'], ...
           };
% savePath = datasetPath;

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

        dataset.setReferenceCenterMethod('forEachContour');
        dataset.calculatePolarCoordinates();

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
    
    %%%
%     save([savePath,'/',name,'_center_of_each_mode.mat'],'dataset');
    
%     % create version with averaged center of profiles
%     dataset.setReferenceCenterMethod('meanCenter');
%     dataset.calculatePolarCoordinates();
%     dataset.calculateFourierTransform();
%     save([savePath,'/',name,'_averaged_center.mat'],'dataset');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

return
%%
close all;
clear all;
clear classes;

fourierSeriesFilenameEachCenter = 'fourierSeriesCenterOfEachContourCorrected.mat';
fourierSeriesFilenameMeanCenter = 'fourierSeriesMeanCenterCorrected.mat';

basePath = {...
            'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/examples/'; ...
            'C:/Private/PhD_Publications/Publication_of_Algorithm/Code/examples/'; ...
%             '/media/lisa_3/figure_10_analysis_mike_2016-05-05/guv_analysis/popc/2014-06-03/'; ...
            };

basePaths = {...
             basePath{1}, ...
             basePath{2}, ...
             };
        
datasetPath = { ...
               [basePath{1},'/','guv_popc/tracking/tracking_000/'], ...
               [basePath{2},'/','rbc/tracking/tracking_000/'], ...
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
indexes = [1:2];

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
    fileName = [savePath{datasetIndex},'/',datasetLabel{datasetIndex},fourierSeriesFilenameEachCenter];
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
    
    fileName = [savePath{datasetIndex},'/',datasetLabel{datasetIndex},fourierSeriesFilenameMeanCenter];
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

