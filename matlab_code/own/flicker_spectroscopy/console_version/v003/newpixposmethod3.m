function [newpixpos,methodinternaldirection]=newpixposmethod3(mempos,pixpos,i)

% Method 3:
% This function finds the next pixel by calculating the angles between the
% vector d spanned by the membrane positions (x_i,y_i) and 
% (x_{i-10},y_{i-10}) and the coordinate axis x and y & v and w of the two
% coordinate systems and checking if these are smaller than 22,5°. If this
% is the case for one of the axis the new pixel will be chosen in this
% direction and returned as 'newpixpos'

% Constants
% The coordinate axis directions
x=[1;0];
y=[0;1];
v=[1;1];
w=[-1;1];

% Variables:
% mempos: matrix containing the already calculated membrane positions
% pixpos: matrix containing the already calculated pixel positions
% i: index of the current pixel to be calculated
% angle to be calculated between the axis and the vector d

% methodinternaldirection: control variable to check algorithm in what direction has moved

if i>10 % this checks that i is larger than 10 so that d can be calculated

%newpixpos: vector returning the new pixel position

d=(mempos(i,:)-mempos(i-10,:));

if norm(d) ~= 0 
% check angle between x-axis and d
alpha=acos((d*x)/(norm(d)*norm(x)));
if alpha>=0 && alpha<=0.3927
    newpixpos=pixpos(i,:)+transpose(x);
    methodinternaldirection=11;
    return;
elseif alpha>=2.7489 && alpha<=3.1416
    newpixpos=pixpos(i,:)-transpose(x);
    methodinternaldirection=12;
    return;
end

% check angle between y-axis and d (only earlier checks failed)
alpha=acos((d*y)/(norm(d)*norm(y)));
if alpha>=0 && alpha<=0.3927
    newpixpos=pixpos(i,:)+transpose(y);
    methodinternaldirection=21;
    return;
elseif alpha>=2.7489 && alpha<=3.1416
    newpixpos=pixpos(i,:)-transpose(y);
    methodinternaldirection=22;
    return;
end

% check angle between v-axis and d (only earlier checks failed)
alpha=acos((d*v)/(norm(d)*norm(v)));
if alpha>=0 && alpha<=0.3927
    newpixpos=pixpos(i,:)+transpose(v);
    methodinternaldirection=31;
    return;
elseif alpha>=2.7489 && alpha<=3.1416
    newpixpos=pixpos(i,:)-transpose(v);
    methodinternaldirection=32;
    return;
end

% check angle between w-axis and d (only earlier checks failed)
alpha=acos((d*w)/(norm(d)*norm(w)));
if alpha>=0 && alpha<=0.3927
    newpixpos=pixpos(i,:)+transpose(w);
    methodinternaldirection=41;
    return;
elseif alpha>=2.7489 && alpha<=3.1416
    newpixpos=pixpos(i,:)-transpose(w);
    methodinternaldirection=42;
    return;
end
end
end

% if none of the cases tested above apply the function gives back the pixel
% postion (x_i,y_i), which will give a 'FALSE' of 'pixelpositiontest'
% function, prompting the main program to try the next newpixmethod-function
newpixpos=pixpos(i,:);
methodinternaldirection=0;