function [logicalresponse]=pixelpositiontest(pixpos,pixeltestparameter,i)

% Paramters:
% pixeltestparameter: is the angle parameter for the second pixel test.

% Variables:
% pixpos: vector containing the pixelpositions
% i:index of the current pixel (equals the index of the loop in which the functions will be used)
% logicalresponse: if all tests go well this value will be returned as 1
% (TRUE). If not it will be 0 (FALSE)

% this function performs the three tests on the new pixel-position i+1 and
% gives back a FALSE (=0) in case one of the fails

logicalresponse=1; %if one of the tests below fails it will set the logicalresponse to 0 (i.e.: FALSE)

% Test 1:
% Check whether the new pixel position pixpos (i+1,:) is different from the anterior one
% pixpos (i,:)

if (pixpos(i+1,:)==pixpos(i,:))
    logicalresponse=0;
    return;
end


%Test 2:
%???correctly described in the Thesis???
%calculate the vector between (x_i,y_i) and the point to be tested
%(x_i+1,y_i+1) and the vector between (x_i-1,y_i-1) and (x_i,y_i) both on
%the pixel-grid. The absolute value between the two vectors is calculated
%and compared with empericly determined parameter. If it is superior to
%this parameter the test fails

if i>1
t=pixpos(i+1,:)-pixpos(i,:);
t_0=pixpos(i-1,:)-pixpos(i,:);

if pixeltestparameter>abs(acos((t*transpose(t_0))/(norm(t)*norm(t_0))))
    logicalresponse=0;
    return;
end
end
