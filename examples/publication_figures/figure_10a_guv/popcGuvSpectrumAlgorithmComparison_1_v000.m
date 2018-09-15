function popcGuvSpectrumAlgorithmComparison_1_v000

%%
% run('/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/setPaths.m');
%run('../../../matlab_code/setPaths.m');
%addpath('../../../matlab_code/setPaths.m');
pathBase = '../../../matlab_code/';
addpath([pathBase,'/own/data_analysis/flickeringDataClass']);
addpath([pathBase,'/external/hline_vline']);
addpath([pathBase,'/own/figureSettings']);

%%
% close all;

%%
clear all;

%%
scriptName = mfilename;
scriptDataDir = [scriptName,'_data'];
mkdir(scriptDataDir);

%%
% datasetPaths = { ...
%                 '/media/data_volume/non-mirrored_files/work/phd_thesis/flicker_spectroscopy_data/popc/2014-06-03/vesicle_4/movie_1_tracking/matlab_tracking_001'; ...
%                 '/media/data_volume/non-mirrored_files/work/phd_thesis/flicker_spectroscopy_data/popc/2014-06-03/vesicle_4/movie_1_tracking/tracking_001/'; ...
%                 };
datasetPaths = { ...
                '../../guv/tracking/matlab_tracking_002/'; ...
%                 'C:\Private\PhD_Publications\Publication_of_Algorithm\Data\popc\2014-06-03\vesicle_4\movie_1_tracking_from_Lisa_2018-06-10\movie_1_tracking/matlab_tracking_002'; ...
                '../../guv/tracking/tracking_000/'; ...
                };

fourierSeriesDataFilename = 'fourierSeriesCenterOfEachContourCorrected.mat';
radiusSeriesDataFilename = 'radiusSeries.mat';

datasetName =  {...
                'popc_2014-06-03_vesicle_4_movie_1_matlab_tracking_002'; ...
                'popc_2014-06-03_vesicle_4_movie_1_tracking_001'; ...
              };

datasetFps = [...
            1500; ...
            1500; ...
            ];

assert(length(datasetName)==length(datasetPaths));
assert(length(datasetFps)==length(datasetPaths));

%%
selectedDatasetIndexes = [1,2];

%%
% selectModeNrs = (1:1:50);
% plotColors = varycolor(length(selectModeNrs));

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

        
        %%
        load([datasetPaths{datasetIndex},'/',fourierSeriesDataFilename]);
        dataset(datasetIndex).fourierseries = fourierSeries;
        load([datasetPaths{datasetIndex},'/',radiusSeriesDataFilename]);
        dataset(datasetIndex).radiusseries = radiusSeries;
        
        %%
%         keyboard
        %%
        meanRadius = nanmean(dataset(datasetIndex).radiusseries);
        
        c_abs = abs(dataset(datasetIndex).fourierseries);
%         spectrum = ((meanRadius^3 * pi)/2) * (nanmean(c_abs.^2,2) - nanmean(c_abs,2).^2); % [eq.14,pecreaux2004]
        spectrumDataNonNormalized = (nanmean(c_abs.^2,2) - nanmean(c_abs,2).^2);
        
        %%
        wavenumber = (1:length(spectrumDataNonNormalized))/meanRadius;
        
        datasetsResults{datasetIndex}.fps = datasetFps(datasetIndex);
        datasetsResults{datasetIndex}.spectrumDataNonNormalized = spectrumDataNonNormalized;
        datasetsResults{datasetIndex}.wavenumber = wavenumber;
        datasetsResults{datasetIndex}.meanRadius = meanRadius;
        datasetsResults{datasetIndex}.radiusSeries = radiusSeries;

        datasetsResults{datasetIndex}.datasetName = datasetName{datasetIndex};
        
        datasetResult = datasetsResults{datasetIndex};
        
        save(fileName,'datasetResult');
        
        clear tmp, dataset;
    end
end

%% figure settings
% load figure settings
%figureSettings;
% fontSizeMainAxes = 7;
set(0, 'defaulttextfontsize', 32);  % title
set(0, 'defaultaxesfontsize', 24);  % axes labels
set(0, 'defaultlinelinewidth', 2);

%%
close(figure(1));
figure(1);

%set(gca,'FontSize',fontSizeMainAxes);

datasetLabels = {'Pecreaux'' algorithm','HiPFSTA'};
% datasetLabelsFrags = {'Pecreaux','HiPFSTA'};
% datasetPlotStyles = {'k--','k-.','k:','k-','k--o','k-.square'};
datasetPlotColor = {'b','r','b','c','m','k','y','r','g'};
% datasetPlotFitStyle = {'-','-','-','-','-','-','-','-','-'};
datasetPlotDataStyle = {'o','s','o','o','o','o','o','s','s'};

counter = 1;
hold all;
for datasetIndex = selectedDatasetIndexes
%     meanRadius = nanmean(dataset(datasetIndex).radiusseries);
    meanRadius = datasetsResults{datasetIndex}.meanRadius;
    spectrumData = ((meanRadius^3 * pi)/2) * datasetsResults{datasetIndex}.spectrumDataNonNormalized;
    wavenumber = datasetsResults{datasetIndex}.wavenumber;

    datasetDataPlotStyles = [datasetPlotColor{counter},datasetPlotDataStyle{counter}];
    plot(wavenumber,spectrumData,datasetDataPlotStyles,'displayname',datasetLabels{counter});
%     plot((1:length(spectrumData)),spectrumData,datasetDataPlotStyles,'displayname',datasetLabels{counter});
    
    counter = counter + 1;
end
hold off;

hold on;
% xScale = 0.22;
xScale = 0.24;
x = linspace(0,2*pi*xScale,length(wavenumber));
% prefactor = 2.646e-26;
prefactor = 7.226e-25;
% plot(wavenumbersTwoSided,prefactor*abs(sinc(x)),'r')
% sinc2fnc = prefactor*abs(sinc(x).^2);
% plot(wavenumber,sinc2fnc,'k-.','displayname','$\sinc^2$')
% plot(prefactor*abs(sinc(x).^3),'b')
sinc5fnc = prefactor*abs(sinc(x).^5);
% wavenumber = wavenumber / 1.45;
plot(wavenumber,sinc5fnc,'k--','displayname','noise floor','LineWidth',2);
% plot(prefactor*abs(sinc(x).^5),'m')
hold off;


% xlim([1.5e5,1.5e8]);
% % ylim([1e-25,1e-19]);
% ylim([1e-28,1e-19]);

% xlim([0.0341e+07,5.4923e+07]);
% xlim([1.6e+05,10.4923e+07]);
% ylim([1e-24,0.7049e-19]);

% xlim([1.6e+05,10.4923e+07]);
% ylim([5e-24,0.7049e-19]);
% format figure
set(gca,'xscale','log');
set(gca,'yscale','log');

pixelSize = 50e-9;
q_pixel = 2*pi/pixelSize;
[hLine,hLabel] = vline(q_pixel,'--k','q_{pix}');
set(hLine,'LineWidth',2);
pos = get(hLabel,'position');
% pos(1) = 1.3e8;
pos(1) = 8e7;
pos(2) = 2e-21;
set(hLabel,'position',pos);

xlim(gca,[0.0017e8,2e8]);
ylim(gca,[1.0e-28,1e-20]);


% pixelSize = 50e-9;
% [hhh,hLabelText]=vline(2*pi/pixelSize,'k--','$1\px$'); % plot vline at wavenumber corresponding to pixel size
% set(hLabelText,'FontSize',fontSizeMainAxes)

%box on;
%
%xlabel('q [m^{-1}]');
%ylabel('P(q) [m^3]');
% set(get(gca,'ylabel'),'position',[relLocX(-0.16),relLocY(0.5),0])
% set(get(gca,'xlabel'),'position',[relLocX(0.5),relLocY(-0.1),0])

%hLeg = legend('-dynamiclegend');
% hLegc = get(hLeg,'children');
% for index = 1:length(datasetLabels)
%     set( findobj(hLegc,'string',datasetLabels{index}), 'userdata',...
%     ['matlabfrag:',datasetLabelsFrags{index}]);
% end
%set(hLeg,'location','southwest');

% ylim([0.5e-24,5e-20]);

%%%
%return
%
%%%
%saveFile = [scriptName,'/',scriptName,'_subfig1'];
%setFigWidth('width','twocolumn');
%
%%%
%matlabfrag(saveFile);
%
%%% functions
%function result = theoreticalSpectrum(wavenumber,temperature,bendingRigidity,membraneTension)
%    % these functions calculate the theoretical PSD in real-space according to
%    % [Betz2009]
%    kBoltzmann = 1.3806e-23;
%    kT = kBoltzmann*temperature;
%    sigma = membraneTension;
%    kappa = bendingRigidity;
%    q = wavenumber;
%
%    % result = ( 1 / (pi*meanRadius^3) ) * (kT/sigma) * ( (1./q) - 1./sqrt( (sigma/kappa) + q.^2 ) );
%    result = (kT/(2*sigma)) * ( (1./q) - 1./sqrt( (sigma/kappa) + q.^2 ) );
%
%
%function r0 = calculateRadius(angles,radii)
%%                 angles = mempos(:,2);
%%                 radii = mempos(:,1);
%    [angles,indexes] = sort(angles);
%    radii = radii(indexes);
%
%    tmp3 = sum( ( radii(2:end) + radii(1:end-1) ) .* abs(angles(1:end-1) - angles(2:end)) );
%    tmp3 = tmp3 + ( radii(end) + radii(1) ) .* abs(angles(end) + angles(1)); % we use '+' between the angles, because the angle changes signs here
%    r0 = (1/(4*pi)) * tmp3;