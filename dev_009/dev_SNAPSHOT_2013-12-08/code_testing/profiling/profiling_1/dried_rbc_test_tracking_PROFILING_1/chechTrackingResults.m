run('/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/setPaths.m');

%%
clear all;

%%
load('contourCoordinatesX.mat')
load('contourCoordinatesY.mat')
load('contourCenterCoordinatesX.mat')
load('contourCenterCoordinatesY.mat')

%%
close(figure(1));
figure(1);

hold all;
for ind = 1:10:2750
%     plot(contourCoordinatesX(:,ind)-contourCenterCoordinatesX(ind),contourCoordinatesY(:,ind)-contourCenterCoordinatesY(ind)); daspect([1 1 1])
    plot(contourCoordinatesX(:,ind),contourCoordinatesY(:,ind)); daspect([1 1 1])
end
hold off;

