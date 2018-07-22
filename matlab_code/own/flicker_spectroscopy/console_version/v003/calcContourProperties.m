function [contourCenter,contourCircumference,contourRadius] = calcContourProperties(xymempos)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% calcContourProperties   This function calculates the properties of the
%%% tracked contour given the membrane coordinates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[contourCircumference,sumds] = calcContourCircumference(xymempos);
                               
contourCenter = calcContourCenter(xymempos,contourCircumference,sumds);

contourRadius = calcContourRadius(contourCircumference);