function [newpixpos,methodinternaldirection]=newpixposmethod4(mempos,pixpos,i,methode4parameter)

% Method 4:
% This function finds next pixel by calculating the vector spanned between
% the membrane positions (x_(i-10),y_(i-10)) and (x_i,y_i). If the
% x-abciss (the y-abciss respectively) is sufficiently large (or small
% if negative) it will choose the new pixel position in that direction and
% give it back as the variable newpixpos.

%variables:
xychange=zeros(1:2); % change variable to that gives back the change to pixpos(i,:) to get newpixpos
methodinternaldirection=0; % variable giving which method was used inside
                           % the newpixposmethod4 algorithm; for checking purposes

% remark to errorchecking:
% methodinternaldirection    directionchange
%       1                       (1,0)
%       2                       (-1,0)
%       4                       (0,1)
%       5                       (1,1)
%       6                       (-1,1)
%       8                       (0,-1)
%       9                       (1,-1)
%       10                      (-1,-1)


if i>10
   xyabciss=mempos(i,:)-mempos(i-10,:);
   
   % check x-abciss of the spanned vector between membrane positions
   % (x_(i-10),y_(i-10)) and (x_i,y_i) change pixpos(i+1,1) (i.e.: the 
   % x-position) accordingly for newpixpos
   if xyabciss(1)>methode4parameter
       xychange(1)=1;
       methodinternaldirection=1;
   elseif xyabciss(1)<-methode4parameter
       xychange(1)=-1;
       methodinternaldirection=2;
   end
   
   % check y-abciss of the spanned vector between membrane positions
   % (x_(i-10),y_(i-10)) and (x_i,y_i) change pixpos(i+1,2) (i.e.: the
   % y-position) accordingly for newpixpos
   if xyabciss(2)>methode4parameter
       xychange(2)=1;
       methodinternaldirection=methodinternaldirection+4;
   elseif xyabciss(2)<-methode4parameter
       xychange(2)=-1;
       methodinternaldirection=methodinternaldirection+8;
   end
end

% Gives back the new pixelposition newpixpos. If the variable 'xychange' is
% unchange 0 this will give back the the pixel position pixpos(i,:) for
% pixpos(i+1,:) causing the pixelpositiontest function of the main program
% to continue with the next newpixposmethod

newpixpos=pixpos(i,:)+xychange;