function pixpos = pixelposition(intensities,startpoint,endpoint,linfitparameter)

%This function calculates the pixelposition of the membrane in the chosen
%direction (i.e.: x- or y- or other direction) detiremined by the
%input vector containing the intesities

%output:
%pixpos: coordinate of the membrane position returned by this function
%m: slope of linear fit straight

%imput:
%intensities: vector containing the intensities of one of the lines of the
%image (one column/row of image)

incline = 0;
for i = startpoint:endpoint
    p = polyfit(transpose(i-linfitparameter:i+linfitparameter),intensities(i-linfitparameter:i+linfitparameter),1);
    if (abs(p(1))>abs(incline(1)))
        incline=p(1);
        pixpos=i; %basepixel of linear fit with greatest slope
    end
end

