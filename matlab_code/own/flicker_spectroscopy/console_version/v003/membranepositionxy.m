function [slope,mempos,interceptpoint,meanvalue]=membranepositionxy(intensities,linfitparameter)

%determines membraneposition in the x-y-coordinate system

%this function determines the membrane position one dimension (i.e.: the direction of the intensities vector) by calculation of
%the intercept point of the linear fit straight through a certain pixel
%position and the mean intesity of a number of pixels around this pixel position both performed on the "intensities" vector given to the function;

%it returns the exact position of the membrane the membrane position and
%the value of the slope of the linear fit straight

%Input:
%intesities: vector containing the intensities over which will be fitted
%pixpos: pixel position around which the linear fit will be performed
%linfitparameter: half of the number of points around [pixpos] over which the linear fit
%will be performed
%meanparameter: half of the number of points around [pixpos] over which the
%mean intensity will be calculated

%Output:
%mempos: koordinate of the membrane position
%slope: slope of the linear fit straight

%determine center of intensitiesvector
centerelement=round(length(intensities)/2);

p=polyfit((-linfitparameter:linfitparameter),transpose(intensities(centerelement-linfitparameter:centerelement+linfitparameter)),1); %calculate linear fit straight
meanvalue=mean(intensities);

mempos=(meanvalue-p(2))/p(1);
slope=p(1);
interceptpoint=p(2);