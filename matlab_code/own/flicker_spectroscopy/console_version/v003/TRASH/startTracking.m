%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% startTracking
%%% This is wrapper-script for initializing and starting the tracking
%%% process.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% setup path to program-files
programPath = '/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/own/flicker_spectroscopy/code_reduced/';
% programPath = '/media/data_volume/mirrored_files/work/phd_thesis/programming/flicker_spectroscopy/code/';
addpath(programPath);

% set name of config-file
configFileName = 'config.ini';

% get locations for possible 
defaultConfigFilePath = [programPath,'/config'];
workingPath = pwd;

if exist([workingPath,'/config.ini'],'file') == 2
    configFilePath = workingPath; % check if a config file exists in the current directory
else
    configFilePath = defaultConfigFilePath;
    configFileName = 'default_config.ini';
end

% %%
% % load settings
% % trackingParameters = loadSettings(configFilePath,configFileName);
% trackingParameters = loadToStructure(configFilePath,configFileName);
% 
% % keyboard
% %%
% % dynamicaly adjusted parameters
% % calculate the the corresponding parameters for the diagonal calculations
% trackingParameters.diagLinFitParameter = ceil(trackingParameters.linfitparameter/1.4142);
% trackingParameters.diagMeanParameter = ceil(trackingParameters.meanparameter/1.4142);
% trackingParameters.displaystartcounter = trackingParameters.displaystart;
% 
% %%
% % load program parameters
% % programParameters = loadSettings(configFilePath,'programParameters.config');
% programParameters = loadToStructure(configFilePath,configFileName);
% 
% % programParameters = getNumberOfImages(programParameters);

% configFilePath = constructConfigPath;
configFileName = 'default_config.ini';
handles.parameterStruct = loadToStructure(configFilePath,configFileName);

%% creat directory for saving the tracked contours
% createContoursDirectory(programParameters);

%%
% start tracking
% tracking(trackingParameters,programParameters);
parameterStruct = handles.parameterStruct;
[programParameters,trackingParameters,trackingVariables] = initializeTracking(parameterStruct);
handles.trackingVariables = trackingVariables;
% setup up file system information
programParameters = getNumberOfImages(programParameters);

% trackingVariables.contourRadius = handles.trackingVariables.contourRadius;
% trackingVariables.startingPixelPosition = handles.trackingVariables.startingPixelPosition;
% trackingVariables = handles.trackingVariables;

console_tracking(handles);