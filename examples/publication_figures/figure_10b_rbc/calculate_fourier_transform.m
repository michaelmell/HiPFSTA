%% setup environment
close all;
clear all;

pathBase = '../../../matlab_code/';
addpath([pathBase,'/own/data_analysis/flickeringDataClass']);

%% Process RBC data from Pecreaux' algorithm
basePath = '../../';

datasetPath = {'rbc/tracking/matlab_tracking/'};
datasetLabelAlternative = {'rbc_healthy_2014-05-05_rbc_4_movie_1_tracking_matlab_tracking_002'};

indices = 1;
savePath = {[basePath,'/',datasetPath{1}]};

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
    dataset.calculateFourierTransformNEW2();
    fourierSeries = dataset.fourierseries;
    save([savePath{index},'/',fourierSeriesFilenameEachCenter],'fourierSeries','-mat');
    
    dataset.setReferenceCenterMethod('meanCenter');
    dataset.calculatePolarCoordinates();
    dataset.calculateFourierTransformNEW2();
    fourierSeries = dataset.fourierseries;
    save([savePath{index},'/',fourierSeriesFilenameMeanCenter],'fourierSeries','-mat');

    radiusSeries = dataset(1).getRadiusSeries;

    save([savePath{index},'/radiusSeries','.mat'],'radiusSeries','-mat');
    circumferenceSeries = dataset(1).getCircumferenceSeries;
    save([savePath{index},'/circumferenceSeries','.mat'],'circumferenceSeries','-mat');
end

%% Process RBC data from HiPFSTA
clear all;

fourierSeriesFilenameEachCenter = 'fourierSeriesCenterOfEachContourCorrected.mat';
fourierSeriesFilenameMeanCenter = 'fourierSeriesMeanCenterCorrected.mat';

basePath = {'../../';};

basePaths = {basePath{1}};
        
datasetPath = {[basePath{1},'/','rbc/tracking/python_tracking/']};

% check that paths exist before starting
for datasetIndex = 1:length(datasetPath)
    if ~exist(datasetPath{datasetIndex},'dir')
        error(['Dataset does not exist: ',char(10),datasetPath{datasetIndex}])
    end
end
           
savePath = datasetPath;

mainName = '';
datasetLabel = {};
for index = 1:length(datasetPath)
    suffix = strrep(datasetPath{index},[basePaths{index},'/'],'');
    suffix = strrep(suffix,'/','_');
    datasetLabel{index} = [mainName,suffix];
end

indexes = [1];

for datasetIndex = indexes
    % set processing parameters
    dataset = flickeringDataClass();
    dataset.setIdString(datasetLabel{datasetIndex});
    dataset.setLabelString(datasetLabel{datasetIndex});
    dataset.setResolution(50.0e-09);
    dataset.setNrOfModes(1024);
    
    dataset.loadPythonTrackingContours_v000( datasetPath{datasetIndex} );
    dataset.calculateCircumference();
    dataset.calculateBarioCenter();

    fileName = [savePath{datasetIndex},'/',fourierSeriesFilenameEachCenter];

    dataset.setReferenceCenterMethod('forEachContour');
    dataset.calculatePolarCoordinates();
    dataset.calculateFourierTransformNEW2();
    fourierSeries = dataset.fourierseries;
    save(fileName,'fourierSeries','-mat');
    
    fileName = [savePath{datasetIndex},'/',fourierSeriesFilenameMeanCenter];
    dataset.setReferenceCenterMethod('meanCenter');
    dataset.calculatePolarCoordinates();
    dataset.calculateFourierTransformNEW2();
    fourierSeries = dataset.fourierseries;
    save(fileName,'fourierSeries','-mat');
    radiusSeries = dataset(1).getRadiusSeries;

    save([savePath{index},'/radiusSeries','.mat'],'radiusSeries','-mat');
    circumferenceSeries = dataset(1).getCircumferenceSeries;
    save([savePath{index},'/circumferenceSeries','.mat'],'circumferenceSeries','-mat');
end

