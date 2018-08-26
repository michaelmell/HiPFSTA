function center = calcContourCenter(xymempos,vesicleCircumference,sumds)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% calcContourCenter   Function for calculating the circumference of the
%%% vesicle contour
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

center(1) = (1/(2*vesicleCircumference))*xymempos(:,1)'*sumds;
center(2) = (1/(2*vesicleCircumference))*xymempos(:,2)'*sumds;