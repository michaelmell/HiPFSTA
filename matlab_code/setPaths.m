% pathBase = '/home/micha/tmp/work/matlab_functions/';
pathBase = '.';
%pathBase = '../../../matlab_code/';

% % add path to opencl toolbox
% addpath(genpath([pathBase,'/external/opencl_toolbox/opencl-toolbox']));
% 
% 
% % add external functions; e.g. function that were obtained from MATLAB file exchange or other
% addpath([pathBase,'/external']);
% addpath([pathBase,'/external/htmlTables']);
% addpath([pathBase,'/external/matc']);
% addpath([pathBase,'/external/secs2hms']);
% addpath([pathBase,'/external/matlab2tikz/src']);
% addpath([pathBase,'/external/peakdet']);
% addpath([pathBase,'/external/varycolor']);
% addpath([pathBase,'/external/nlparci_new']);
% addpath([pathBase,'/external/latextable']);
% addpath([pathBase,'/external/KOJA/KOJA/koja-package']);
% addpath([pathBase,'/external/circ']);
% 
% % functions used for calculating physics stuff
% addpath([pathBase,'/own']);
% addpath([pathBase,'/own/physics']);
% addpath([pathBase,'/own/physics/seifertTheory']);
% addpath([pathBase,'/own/physics/helfrichMembrane']);
% addpath([pathBase,'/own/physics/zilmanGranekTheory']);
% addpath([pathBase,'/own/physics/arriagaAmplitudeTheory']);
% addpath([pathBase,'/own/physics/translationRelaxation']);
% 
% % functions analyzing data
% addpath([pathBase,'/own/data_analysis/traceAnalyzationClass']);
% addpath([pathBase,'/own/data_analysis/timeTraceClass']);
% addpath([pathBase,'/own/data_analysis/traceSetClass']);
% 
% % utility functions
% addpath(genpath([pathBase,'/own/utilities']));
% 
% % functions for simulation
% addpath([pathBase,'/own/data_analysis/simulationLoaderClass']);
% 
% % function for loading experimental data
% addpath([pathBase,'/own/data_analysis/flickeringLoaderClass']);
addpath([pathBase,'/own/data_analysis/flickeringDataClass']);
% 
% % functions for data loading, saving and caching (e.g.: memory mapping, etc.)
% addpath(genpath([pathBase,'/external/waterloo']));
% addpath(genpath([pathBase,'/external/hlp_microcache']));
% addpath(genpath([pathBase,'/own/utilities/diskcache']));
% addpath(genpath([pathBase,'/external/DataHash']));
% 
% % add HOSA toolbox
% addpath(genpath([pathBase,'/external/HOSA/hosa_d']));
% 
% % fitting tools
% addpath([pathBase,'/toolboxes/ezyfit/ezyfit']);
% addpath([pathBase,'/own/data_fitting/multiscalar_fitting']);
% 
% functions for plotting and saving figures
addpath([pathBase,'/external/hline_vline']);
% addpath([pathBase,'/external/matlabfrag']);
% addpath([pathBase,'/external/errorbarlogx']);
% addpath([pathBase,'/external/errorbar_tick']);
% addpath([pathBase,'/external/ploterr']);
% addpath([pathBase,'/external/SubAxis']);
% addpath([pathBase,'/external/cloudPlot/cloudPlot_upload_package']);
% addpath([pathBase,'/external/suplabel']);
% addpath([pathBase,'/external/savefig']);
% addpath([pathBase,'/external/smoothhist2D']);
% addpath([pathBase,'/external/shadedErrorBar/shadedErrorBar']);
% addpath([pathBase,'/external/polar2']);
% addpath([pathBase,'/external/freezeColors/freezeColors_v23_cbfreeze/freezeColors']);
% addpath([pathBase,'/external/scalebar']);
% addpath([pathBase,'/external/export_fig/export_fig']);
% addpath([pathBase,'/external/shadePlotForEmphasis']);
% addpath([pathBase,'/external/hatchfill/hatchfillpkg']);
% addpath([pathBase,'/external/barwitherr']);

% add settings of figures
addpath([pathBase,'/own/figureSettings']);

% % add functions for converting units
% addpath([pathBase,'/external/Degree_Radian_Conversion'])
% 
% % add function for statistics analysis
% addpath([pathBase,'/external/stbl_code/STBL_CODE'])
% 
% % geometry related functions
% addpath([pathBase,'/external/linecurvature_version1b'])
% addpath([pathBase,'/external/interX/InterX'])
% addpath([pathBase,'/external/FastLineSegmentIntersection/FastLineSegmentIntersection'])
% 
% % power spectrum estimation
% addpath([pathBase,'/external/lpsd'])
% 
% % power spectrum estimation
% addpath([pathBase,'/external/legendreP2/legendreP2'])
% 
