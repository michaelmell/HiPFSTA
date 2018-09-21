%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Runs Matlab implementation of Pecreaux' algorithm to track contours in the 
%%% RBC dataset. The algorithm is published in DOI: 10.1140/epje/i2004-10001-9
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
restoredefaultpath;
addpath('../../../matlab_code/own/flicker_spectroscopy/console_version/v003');
trackImages('matlab_tracking.conf');