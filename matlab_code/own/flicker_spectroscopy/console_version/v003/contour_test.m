function void = contour_test(parameterStruct)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% testcontour   This program checks if the tracked contours were detected
%%% correctly
%%% It uses two methods to do this:
%%% Method 1:
%%% This method checks, that the membrane was circularly closed by comparing
%%% the vector of the 10 (or so) points of the membrane agaisnt that between the
%%% first 10 points. If the angle between them is more than the value given
%%% by the parameter "maxangle" the membrane will be deleted.
%%% 
%%% Method 2:
%%% This method checks that the length of the circumferences of a detected
%%% contour is the same to a cetain percentage as the mean length of all the
%%% circumferences. If this is not the case, the contour will be deleted. The 
%%% maximum deviation from the mean value is given by the parameter "maxlengthdeviation". 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% set program parameters
programParameters = setProgramParameters(parameterStruct);

% setup up file system information
programParameters = getNumberOfImages(programParameters);

%%
% create path were to save the results
pathToResultsDirectory = [programParameters.data_analysis_directory_path,'/',programParameters.data_analysis_configFileName];

% check number of tracked contour files in the contour directory
contournr = 1;
while exist([pathToResultsDirectory,'/contours/contour',num2str(contournr),'.txt']) == 2
   contournr = contournr + 1; 
end
contournr = contournr - 1;

%%
% output of parameters
parameterStruct

%%
% set variables
L = zeros(1,contournr); % vector containing the circumferences of the different contours
deletedcontours(1) = 0; % vector in which the number of the deleted contours is saved

% counter
deletedcontourscounter = 1; % counter to augment by 1 if counter is deleted;
                          % used together with "deletedcontours" to save
                          % the number of the deleted contour
                          
savedcontours(1) = 0;
savedcontourscounter = 1;

% load the index that contains which contours where closed during tracking
contourclosedindex = load([pathToResultsDirectory,'/results/contourclosedindex.txt']);



%%
% Method 2:
% calculate length of circumference

% NOTE: only the circumference of contours not rejected by method 1 will be
% calculated

% variables:
for j = savedcontours
    mempos = load([pathToResultsDirectory,'/contours/contour',num2str(j),'.txt']); % creates nxn-matrix of grayscale-image with intensity values
    
    memposlength = length(mempos); % length of the current mempos vector (i.e: number of contour points)
    ds = zeros(memposlength,1); % vector containing the distance between neighbooring membrane points

for i = 1:memposlength - 1
   ds(i + 1) = sqrt((mempos(i + 1,1) - mempos(i,1))^2 + (mempos(i + 1,2) - mempos(i,2))^2);
   L(j) = L(j) + ds(i);
   if i == memposlength - 1
       ds(1) = sqrt((mempos(1,1) - mempos(i + 1,1))^2 + (mempos(i,2) - mempos(i + 1,2))^2); 
       % to get the last part between the end and starting point of tracked contour
       L(j) = L(j) + ds(1);
   end
end
end



% check whether the vector 'savedcontours' is empty
if length(savedcontours) == 0
    errordlg('All contours were deleted. To solve this try decreasing the value of the ''Contour Circumference'' or increasing ''Maximal Circumference Deviation'' in ''Contour Test Parameters'' ','All contours deleted')
    return;
end

% calculate mean value of the contourlengths
meancircumference = sum(L)/length(savedcontours); % mean length of all the circum ferences
onepercentofmeancircumference = meancircumference/100;

% check the differet contour lengths against the mean value. If the
% deviation in percent is greater than the deviation given by the parameter
% 'parameterStruct.maxlengthdeviation' the contour will be eliminated

tmp = savedcontours; % the temp is used, because were changing savedcontours inside the loop
for j = tmp
    if abs(L(j) - meancircumference)>onepercentofmeancircumference*parameterStruct.maxlengthdeviation
        % delete contour-index from 'savecondtours' vector    
        [m,n] = find(savedcontours == j);
        savedcontours(n) = [];
    
        % write contour-index to 'deletedcontours' vector        
        deletedcontours(deletedcontourscounter) = j;
        method(deletedcontourscounter) = 2;
        deletedcontourscounter = deletedcontourscounter + 1;
    end
end

% this saves the index of the deleted contours
save([pathToResultsDirectory,'/results/deletedcontours.txt'],'deletedcontours','-ascii')

% this saves the index of the saved contours
save([pathToResultsDirectory,'/results/savedcontours.txt'],'savedcontours','-ascii')

% save log-file from of 'testcontour' program
fid = fopen([pathToResultsDirectory,'/results/log_testcontour.txt'],'w');
fprintf(fid,'Total Contours: %i \nSaved Contours: %i \nDeleted Contours: %i',contournr,length(find(savedcontours ~= 0)),length(find(deletedcontours ~= 0)));
fclose('all');

% this display a message box displaying the total number of contours and number of saved contours and
% deleted contours

msgbox(sprintf('Total Contours: %i \nSaved Contours: %i \nDeleted Contours: %i',contournr,length(find(savedcontours ~= 0)),length(find(deletedcontours ~= 0))),'Results','none','modal')
