function [xmin,xmax,ymin,ymax] = getVesicleExtension(contour)
    xmin = floor(min(contour(:,2)));
    xmax = ceil(max(contour(:,2)));
    ymin = floor(min(contour(:,1)));
    ymax = ceil(max(contour(:,1)));

    
