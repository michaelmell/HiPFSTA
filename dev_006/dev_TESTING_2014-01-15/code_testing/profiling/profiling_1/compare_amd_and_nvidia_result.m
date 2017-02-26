run('/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/setPaths.m');

%%
clear all;

%%
load('dried_rbc_test_tracking_PROFILING_1_amd/contourCoordinatesX.mat')
load('dried_rbc_test_tracking_PROFILING_1_amd/contourCoordinatesY.mat')
load('dried_rbc_test_tracking_PROFILING_1_amd/contourCenterCoordinatesX.mat')
load('dried_rbc_test_tracking_PROFILING_1_amd/contourCenterCoordinatesY.mat')
contourCoordinatesX_amd = contourCoordinatesX;
contourCoordinatesY_amd = contourCoordinatesY;
contourCenterCoordinatesX_amd = contourCenterCoordinatesX;
contourCenterCoordinatesY_amd = contourCenterCoordinatesY;

load('dried_rbc_test_tracking_PROFILING_1_nvidia/contourCoordinatesX.mat')
load('dried_rbc_test_tracking_PROFILING_1_nvidia/contourCoordinatesY.mat')
load('dried_rbc_test_tracking_PROFILING_1_nvidia/contourCenterCoordinatesX.mat')
load('dried_rbc_test_tracking_PROFILING_1_nvidia/contourCenterCoordinatesY.mat')
contourCoordinatesX_nvidia = contourCoordinatesX;
contourCoordinatesY_nvidia = contourCoordinatesY;
contourCenterCoordinatesX_nvidia = contourCenterCoordinatesX;
contourCenterCoordinatesY_nvidia = contourCenterCoordinatesY;

%%

startIndex = 1;
% startIndex = 24000;
endIndex = 10;

xLimits = [min(contourCoordinatesX(:,startIndex))-10,max(contourCoordinatesX(:,startIndex))+10];
yLimits = [min(contourCoordinatesY(:,startIndex))-10,max(contourCoordinatesY(:,startIndex))+10];
xLimits = xLimits - contourCenterCoordinatesX(startIndex);
yLimits = yLimits - contourCenterCoordinatesY(startIndex);

for ind = startIndex:1:endIndex
    plot(contourCoordinatesX_amd(:,ind)-contourCenterCoordinatesX_amd(ind),contourCoordinatesY_amd(:,ind)-contourCenterCoordinatesY_amd(ind),'b');
    hold on;
    plot(contourCoordinatesX_nvidia(:,ind)-contourCenterCoordinatesX_nvidia(ind),contourCoordinatesY_nvidia(:,ind)-contourCenterCoordinatesY_nvidia(ind),'r');
    hold off;
    set(gca,'ydir','reverse');
    daspect([1 1 1]);
    xlim(xLimits);
    ylim(yLimits);
    text(relLocX(0.05),relLocY(0.95),['Frame: ',num2str(ind)]);
%     pause(0.03);
    pause();
%     keyboard
    xLimits = get(gca,'xlim');
    yLimits = get(gca,'ylim');
end

%%
choseIndex = 1024;
nrOfBins = 100;

hist(contourCoordinatesX(choseIndex,startIndex:endIndex) - mean(contourCoordinatesX(choseIndex-20:choseIndex+10,startIndex:endIndex)),nrOfBins);
% plot(contourCoordinatesX(choseIndex,startIndex:endIndex) - mean(contourCoordinatesX(choseIndex-10:choseIndex+10,startIndex:endIndex)))

