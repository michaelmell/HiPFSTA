run('/media/data_volume/mirrored_files/work/phd_thesis/matlab_functions/setPaths.m');

%%
clear all;

%%
load('contourCoordinatesX.mat')
load('contourCoordinatesY.mat')
load('contourCenterCoordinatesX.mat')
load('contourCenterCoordinatesY.mat')

%%
zeroIndices = find(contourCoordinatesX(1,:)==0);
zeroIndices(1)

ind = 14000; plot(contourCoordinatesX(:,ind),contourCoordinatesY(:,ind)); daspect([1 1 1])

%%
close(figure(1));
figure(1);

% backgroundImagePath = '../../background_1_1_C001H001S0001/background_1_1_C001H001S0001000001.tif';
% imageDirectoryPath = '../../movie_1_C001H001S0001';

backgroundImagePath = '/media/data_volume/non-mirrored_files/work/phd_thesis/flicker_spectroscopy_data/phase_contrast_microscopy/rbc/healthy/2014-03-25/rbc_1/background_1_1_C001H001S0001/background_1_1_C001H001S0001000001.tif';
imageDirectoryPath = '/media/data_volume/non-mirrored_files/work/phd_thesis/flicker_spectroscopy_data/phase_contrast_microscopy/rbc/healthy/2014-03-25/rbc_1/movie_1_C001H001S0001';


bkgrData = double(imread(backgroundImagePath));

startIndex = 1750;
endIndex = 1900;

xLimits = [min(contourCoordinatesX(:,startIndex))-10,max(contourCoordinatesX(:,startIndex))+10];
yLimits = [min(contourCoordinatesY(:,startIndex))-10,max(contourCoordinatesY(:,startIndex))+10];

for ind = startIndex:endIndex
    subaxis(1,2,1);
    bkgrData = double(imread(backgroundImagePath));
    imgData = double(imread([imageDirectoryPath,'/','movie_1_C001H001S0001',num2str(ind,'%06d'),'.tif']));
    imagesc(imgData./bkgrData);
    daspect([1 1 1])
%     set(gca,'clim',[0.8,1.2]);
    
    subaxis(1,2,2);
    plot(contourCoordinatesX(:,ind),contourCoordinatesY(:,ind));
    set(gca,'ydir','reverse');
    daspect([1 1 1]);
    xlim(xLimits);
    ylim(yLimits);
    text(relLocX(0.05),relLocY(0.95),['Frame: ',num2str(ind)]);
%     pause
    if ind == startIndex
        pause;
    else
        pause(0.03);
    end
    xLimits = get(gca,'xlim');
    yLimits = get(gca,'ylim');
end


%%
close(figure(1));
figure(1);

startIndex = 1;
% startIndex = 27295;
endIndex = 36618;


xLimits = [min(contourCoordinatesX(:,startIndex))-10,max(contourCoordinatesX(:,startIndex))+10];
yLimits = [min(contourCoordinatesY(:,startIndex))-10,max(contourCoordinatesY(:,startIndex))+10];
xLimits = xLimits - contourCenterCoordinatesX(startIndex);
yLimits = yLimits - contourCenterCoordinatesY(startIndex);

pause()
for ind = startIndex:1:endIndex
    plot(contourCoordinatesX(:,ind)-contourCenterCoordinatesX(ind),contourCoordinatesY(:,ind)-contourCenterCoordinatesY(ind));
    set(gca,'ydir','reverse');
    daspect([1 1 1]);
    xlim(xLimits);
    ylim(yLimits);
    text(relLocX(0.05),relLocY(0.95),['Frame: ',num2str(ind)]);
    pause(0.1);
%     pause();
%     keyboard
    xLimits = get(gca,'xlim');
    yLimits = get(gca,'ylim');
end

%%
close(figure(2));
figure(2);

startIndex = 1;
% startIndex = 27295;
endIndex = 200;


xLimits = [min(contourCoordinatesX(:,startIndex))-10,max(contourCoordinatesX(:,startIndex))+10];
yLimits = [min(contourCoordinatesY(:,startIndex))-10,max(contourCoordinatesY(:,startIndex))+10];
xLimits = xLimits;
yLimits = yLimits;

pause()
for ind = startIndex:1:endIndex
    plot(contourCoordinatesX(:,ind),contourCoordinatesY(:,ind));
    set(gca,'ydir','reverse');
    daspect([1 1 1]);
    xlim(xLimits);
    ylim(yLimits);
    text(relLocX(0.05),relLocY(0.95),['Frame: ',num2str(ind)]);
    pause(0.3);
%     pause();
%     keyboard
    xLimits = get(gca,'xlim');
    yLimits = get(gca,'ylim');
end

%%
maxInd = 20000;
nrOfContourPoints = size(contourCoordinatesX,1);
xCoordMean = mean(contourCoordinatesX(:,1:maxInd)-repmat(contourCenterCoordinatesX(1:maxInd),nrOfContourPoints,1),2);
yCoordMean = mean(contourCoordinatesY(:,1:maxInd),2);
plot(xCoordMean,yCoordMean);
set(gca,'ydir','reverse');
daspect([1 1 1]);

%%
choseIndex = 1024;
nrOfBins = 100;

hist(contourCoordinatesX(choseIndex,startIndex:endIndex) - mean(contourCoordinatesX(choseIndex-20:choseIndex+10,startIndex:endIndex)),nrOfBins);
% plot(contourCoordinatesX(choseIndex,startIndex:endIndex) - mean(contourCoordinatesX(choseIndex-10:choseIndex+10,startIndex:endIndex)))

%%
choseIndex = 1024;
for ind = startIndex:endIndex
    plot(contourCoordinatesX(choseIndex,ind) - mean(contourCoordinatesX(choseIndex-5:choseIndex+5,ind)),contourCoordinatesY(choseIndex,ind));
    pause;
end

%%
choseIndex = 512;
nrOfBins = 120;

hist(contourCoordinatesY(choseIndex,startIndex:endIndex) - mean(contourCoordinatesY(choseIndex-20:choseIndex+20,startIndex:endIndex)),nrOfBins)
% plot(contourCoordinatesX(choseIndex,startIndex:endIndex) - mean(contourCoordinatesX(choseIndex-10:choseIndex+10,startIndex:endIndex)))

%%
startIndex = 1;
endIndex = 16000;
nrOfBins = 100;

for choseIndex = startIndex:endIndex
    hist(diff(contourCoordinatesX(choseIndex,startIndex:endIndex)),nrOfBins)
    % plot(contourCoordinatesX(choseIndex,startIndex:endIndex) - mean(contourCoordinatesX(choseIndex-10:choseIndex+10,startIndex:endIndex)))
    pause;
end

%%
startIndex = 1;
endIndex = 20000;

hold all;
% for coordIndex = [1:10,1024:1034]
for coordIndex = [1,1024] % [1024:1034]
%     subaxis(2,1,1);
    qqplot(randn(1,20000),contourCoordinatesX(coordIndex,startIndex:endIndex)-mean(contourCoordinatesX(coordIndex,startIndex:endIndex)));
%     subaxis(2,1,2);
%     pause;
end
hold off;
% hold all;
% % for coordIndex = [1:10,1024:1034]
% for coordIndex = [512,1536] % [1024:1034]
% %     subaxis(2,1,1);
%     qqplot(randn(1,20000),contourCoordinatesY(coordIndex,startIndex:endIndex)-mean(contourCoordinatesY(coordIndex,startIndex:endIndex)));
% %     subaxis(2,1,2);
% %     pause;
% end
% hold off;

% hold on;
% qqplot(randn(1,20000),randn(1,20000));
% hold off;

%%
choseIndex = 1024;
qqplot(randn(1,20000),contourCoordinatesX(choseIndex,startIndex:endIndex) - mean(contourCoordinatesX(choseIndex-5:choseIndex+5,startIndex:endIndex),1));

%%
choseIndex = 1;
qqplot(randn(1,20000),diff(contourCoordinatesX(choseIndex,startIndex:endIndex)));
hold on;
choseIndex = 512;
qqplot(randn(1,20000),diff(contourCoordinatesY(choseIndex,startIndex:endIndex)));
% choseIndex = 1024;
% qqplot(randn(1,20000),diff(contourCoordinatesX(choseIndex,startIndex:endIndex)));
hold off;

%%
choseIndex = 11;
cdfplot(contourCoordinatesX(choseIndex,startIndex:endIndex) - mean(contourCoordinatesX(choseIndex,startIndex:endIndex)));
hold on;
cdfplot(contourCoordinatesX(choseIndex,startIndex:endIndex) - mean(contourCoordinatesX(choseIndex-5:choseIndex+5,startIndex:endIndex),1));
cdfplot(randn(20000,1));
hold off;

%%
figure(1)
coordIndex = 1;
qqplot(randn(1,20000),contourCoordinatesX(coordIndex,startIndex:endIndex)-mean(contourCoordinatesX(coordIndex,startIndex:endIndex)));
hold on;
coordIndex = 1024;
qqplot(randn(1,20000),contourCoordinatesX(coordIndex,startIndex:endIndex)-mean(contourCoordinatesX(coordIndex,startIndex:endIndex)));
coordIndex = 512;
qqplot(randn(1,20000),contourCoordinatesY(coordIndex,startIndex:endIndex)-mean(contourCoordinatesY(coordIndex,startIndex:endIndex)));
hold off;
% hold on;
% qqplot(randn(1,20000),contourCoordinatesX(coordIndex,startIndex:endIndex)-contourCenterCoordinatesX(coordIndex));
% hold off;

figure(2)
data1 = contourCoordinatesX(coordIndex,startIndex:endIndex)-mean(contourCenterCoordinatesX(coordIndex));
data2 = contourCoordinatesX(coordIndex,startIndex:endIndex)-contourCenterCoordinatesX(coordIndex);
qqplot(data1,data2);

%%
coordIndex = 1;
data1 = contourCoordinatesX(coordIndex,startIndex:endIndex)-mean(contourCoordinatesX(coordIndex,startIndex:endIndex));
coordIndex = 512;
data2 = contourCoordinatesY(coordIndex,startIndex:endIndex)-mean(contourCoordinatesY(coordIndex,startIndex:endIndex));
qqplot(data1,data2);
