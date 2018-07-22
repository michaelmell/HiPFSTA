function [vesicleCircumference,sumds] = calcContourCircumference(xymempos)

% Variables:
vesicleCircumference = 0; % length of circumference
xymemposlength = length(xymempos); % length of the vector 'xymempos'
ds = zeros(xymemposlength,1); % vector containing the distances between neighbooring membrane positions
sumds = zeros(xymemposlength,1); % vector containing sum between ds(index2) and ds(index2+1)

% calculate length of circumference
for index2 = 1:xymemposlength-1
   ds(index2+1) = sqrt((xymempos(index2+1,1)-xymempos(index2,1))^2+(xymempos(index2+1,2)-xymempos(index2,2))^2);
   vesicleCircumference = vesicleCircumference+ds(index2);
   if index2==xymemposlength-1
       ds(1) = sqrt((xymempos(1,1)-xymempos(index2+1,1))^2+(xymempos(index2,2)-xymempos(index2+1,2))^2); 
       % to get the last part between the end and starting point of tracked contour
       vesicleCircumference = vesicleCircumference+ds(1);
   end
end

% loop for calculating 'sumds'
for index2 = 1:xymemposlength-1
    sumds(index2) = ds(index2)+ds(index2+1);
    if index2==xymemposlength-1
        sumds(index2+1) = ds(index2+1)+ds(1);
    end
end