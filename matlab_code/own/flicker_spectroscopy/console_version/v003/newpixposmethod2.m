function [newpixpos,methodinternaldirection]=newpixposmethod2(pixpos,i,Sx,Sy,Sv,Sw,xdirection,ydirection,slopediff)

% Method 2:
% This function finds the next point by comparing the slopes of the
% different directions (i.e.:x,y,v,w) about the position (x_i,y_i) of the
% pixel grid. If one them is netto larger than the others it will choose
% the point perpendicular to the slop-direction as the next pixel position
% (x_(i+1),y_(i+1)). And gives it back as 'newpixpos'
% The direction of the perpendicular movement is given by the direction of
% the last 10 point of the pixel grid (i.e.: the variables 'xdirection' and
% 'ydirection' of the main program).

% variables:
% methodinternaldirection: variable to check algorithm in what direction has moved

% paramters:
% slopediff: this parameter is the minimal value by which one of the 
% slopes has to be bigger than the others so that it will be chosen; this
% paramter is given to the function by main program


% check Sx against the other slopes and in case move in y-direction
% corresponding to (0,1)
if Sy<Sx-slopediff && Sx<Sw-slopediff && Sx<Sv-slopediff
    newpixpos=[pixpos(i,1),pixpos(i,2)+ydirection];
    methodinternaldirection=1;
    return;
end

% check Sy against the other slopes and in case move in x-direction
% corresponding to (1,0)
if Sx<Sy-slopediff && Sx<Sw-slopediff && Sx<Sv-slopediff
    newpixpos=[pixpos(i,1)+xdirection,pixpos(i,2)];
    methodinternaldirection=2;
    return;
end

% check Sv against the other slopes and in case move in w-direction
% corresponding to (-1,1)
if Sw<Sy-slopediff && Sw<Sx-slopediff && Sw<Sv-slopediff
    newpixpos=[pixpos(i,1)+xdirection,pixpos(i,2)+ydirection];
    methodinternaldirection=3;
    return;
end

% check Sw against the other slopes and in case move in v-direction
% corresponding to (1,1)
if Sv<Sy-slopediff && Sv<Sx-slopediff && Sv<Sw-slopediff
    newpixpos=[pixpos(i,1)+xdirection,pixpos(i,2)+ydirection];
    methodinternaldirection=4;
    return;
end

% if none of the cases tested above apply the function gives back the pixel
% postion (x_i,y_i), which will give a 'FALSE' of 'pixelpositiontest'
% function, prompting the main program to try the next newpixmethod-function
newpixpos=pixpos(i,:);
methodinternaldirection=0;