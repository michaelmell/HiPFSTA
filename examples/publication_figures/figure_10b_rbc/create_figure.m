%% Setup environment
clear all;

pathBase = '../../../matlab_code/';
addpath([pathBase,'/own/data_analysis/flickeringDataClass']);
addpath([pathBase,'/external/hline_vline']);

%% Set dataset that should be processed/shown
selectedDatasetIndexes = [1,2];

datasetPaths = { ...
                '../../rbc/tracking/matlab_tracking_002/'; ...
                '../../rbc/tracking/tracking_000/'; ...
                };

fourierSeriesDataFilename = 'fourierSeriesCenterOfEachContourCorrected.mat';
radiusSeriesDataFilename = 'radiusSeries.mat';

datasetName =  {...
                'rbc_2014-05-05_healthy_movie_1_matlab_tracking_002'; ...
                'rbc_2014-05-05_healthy_movie_1_tracking_001'; ...
              };

datasetFps = [...
            1500; ...
            1500; ...
            ];

assert(length(datasetName)==length(datasetPaths));
assert(length(datasetFps)==length(datasetPaths));

%% Process and create datasets. If it already exists just load them.
scriptName = mfilename;
scriptDataDir = [scriptName,'_data'];
mkdir(scriptDataDir);

for datasetIndex = selectedDatasetIndexes
    fileName = [scriptDataDir,'/',datasetName{datasetIndex},'.mat'];
    if exist(fileName)
        disp(['Loading results from dataset: ',num2str(datasetIndex),' of ',num2str(length(datasetPaths))]);
        tmp = load(fileName);
        datasetsResults{datasetIndex} = tmp.datasetResult;
    else
        disp(['Working on dataset: ',num2str(datasetIndex),' of ',num2str(length(datasetPaths))]);
        disp(['Dataset Name: ']);
        disp(datasetPaths{datasetIndex});
        load([datasetPaths{datasetIndex},'/',fourierSeriesDataFilename]);
        dataset(datasetIndex).fourierseries = fourierSeries;
        load([datasetPaths{datasetIndex},'/',radiusSeriesDataFilename]);
        dataset(datasetIndex).radiusseries = radiusSeries;
        meanRadius = nanmean(dataset(datasetIndex).radiusseries);
        c_abs = abs(dataset(datasetIndex).fourierseries);
        spectrumDataNonNormalized = (nanmean(c_abs.^2,2) - nanmean(c_abs,2).^2);
        wavenumber = (1:length(spectrumDataNonNormalized))/meanRadius;
        datasetsResults{datasetIndex}.fps = datasetFps(datasetIndex);
        datasetsResults{datasetIndex}.spectrumDataNonNormalized = spectrumDataNonNormalized;
        datasetsResults{datasetIndex}.wavenumber = wavenumber;
        datasetsResults{datasetIndex}.meanRadius = meanRadius;
        datasetsResults{datasetIndex}.radiusSeries = radiusSeries;
        datasetsResults{datasetIndex}.datasetName = datasetName{datasetIndex};
        datasetResult = datasetsResults{datasetIndex};
        save(fileName,'datasetResult','-mat');
        clear tmp, dataset;
    end
end

%% Create figure
set(0, 'defaulttextfontsize', 10);  % sets alos size of pixel-indicator label
set(0, 'defaultaxesfontsize', 10);  % axes labels
set(0, 'defaultlinelinewidth', 1);  % linewidth

figure();

datasetLabels = {'Pecreaux'' algorithm','HiPFSTA'};
datasetPlotColor = {'b','r','b','c','m','k','y','r','g'};
datasetPlotDataStyle = {'o','s','o','o','o','o','o','s','s'};

counter = 1;
hold all;

% plot each datasets
for datasetIndex = selectedDatasetIndexes
    meanRadius = datasetsResults{datasetIndex}.meanRadius;
    spectrumData = ((meanRadius^3 * pi)/2) * datasetsResults{datasetIndex}.spectrumDataNonNormalized;
    wavenumber = datasetsResults{datasetIndex}.wavenumber;
    datasetDataPlotStyles = [datasetPlotColor{counter},datasetPlotDataStyle{counter}];
    plot(wavenumber,spectrumData,datasetDataPlotStyles,'displayname',datasetLabels{counter});
    counter = counter + 1;
end
hold off;

% add illustrative noise-floor
hold on;
xScale = 0.31;
x = linspace(0,2*pi*xScale,length(wavenumber));
prefactor = 5.6e-26;
sinc5fnc = prefactor*abs(sinc(x).^5);
plot(wavenumber,sinc5fnc,'k--','displayname','noise floor');
hold off;

% format figure
set(gca,'xscale','log');
set(gca,'yscale','log');

xlim(gca,[0.0017e8,2.7e8]);
ylim(gca,[5.0e-30,1e-20]);

% add indicator for pixel-size wavenumber
pixelSize = 50e-9;
q_pixel = 2*pi/pixelSize;
[hLine,hLabel] = vline(q_pixel,'--k','q_{pix}');
set(hLine,'LineWidth',1);
pos = get(hLabel,'position');
pos(1) = 7e7;
pos(2) = 8e-22;
set(hLabel,'position',pos);

box on;

xlabel('q [m^{-1}]');
ylabel('P(q) [m^3]');

title('RBC spectra comparison');
