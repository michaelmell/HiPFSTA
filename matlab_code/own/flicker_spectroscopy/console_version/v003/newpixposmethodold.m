function [newpixpos,methodinternaldirection,newmempos,Sx,Sy,xbar,ybar]=newpixposmethodold(pixpos,image,xdirection,ydirection,meanparameter,linfitparameter)


% This function uses the first algorithm also used in the program
% "oldmain.m" to find the next pixel position; it is given the current
% pixel position and the matrix image and will return the position of the
% new pixel position


% Variables:
% methodinternaldirection: control variable to check algorithm in what direction has moved


%determine new points

%determine ybar at (x_i+1,y_i) in y-direction
[Sy,ybar]=membranepositionxy(transpose(image(pixpos(1)+xdirection,pixpos(2)-meanparameter:pixpos(2)+meanparameter)),linfitparameter);


%determine xbar at (x_i,y_i+1) in x-direction 
[Sx,xbar]=membranepositionxy(image(pixpos(1)-meanparameter:pixpos(1)+meanparameter,pixpos(2)+ydirection),linfitparameter);

%Sx=abs(Sx);
%Sy=abs(Sy);

if abs(Sy)>abs(Sx)
    %set membrane-position
    mempos(1)=pixpos(1)+xdirection; 
    mempos(2)=pixpos(2)+ybar;
    
    %set new pixel-position
    pixpos(1)=pixpos(1)+xdirection;
    pixpos(2)=pixpos(2);
    
    methodinternaldirection=1;
    
else
    %set membrane-position
    mempos(1)=pixpos(1)+xbar;
    mempos(2)=pixpos(2)+ydirection;
    
    %set new pixel-position
    pixpos(1)=pixpos(1);
    pixpos(2)=pixpos(2)+ydirection;
    
    methodinternaldirection=2;
    
end

newpixpos=pixpos;
newmempos=mempos;
