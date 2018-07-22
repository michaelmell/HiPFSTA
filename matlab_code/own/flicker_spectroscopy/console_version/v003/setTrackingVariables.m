function trackingVariables = setTrackingVariables(trackingParameters,programParameters)

trackingVariables.init = 1; % initialize structure for tracking parameters

% variables:
% cell to which we will write the coordinates for each contour
trackingVariables.contour.coordinates = cell(programParameters.contourNr,1);

% allocation of the matrices that will membrane and pixel position
trackingVariables.pixpos = zeros(trackingParameters.maxnrofpoints,2); % pixel position coordinates
trackingVariables.xymempos = zeros(trackingParameters.maxnrofpoints,2); % membrane position coordinates
trackingVariables.vwmempos = zeros(trackingParameters.maxnrofpoints,2);

% IMPORTANT: The coordinate-system in the following will be chosen so that the x-direction
% corresponds to the row-index of the image-matrix and the y-direction to
% the column-index of the coordinate-system; i.e.: the normal
% right-coordinate-system is turned 90 degrees clockwise

trackingVariables.xymeanmemchange = zeros(trackingParameters.maxnrofpoints,2); % vector containing the mean change in x-/y-direction of the
                                                         % last number of membrane points given by trackingParameters.lastpositions
trackingVariables.vwmeanmemchange = zeros(trackingParameters.maxnrofpoints,2);

% set starting point for membrane tracking; note that the coordinates have
% to be switched
% trackingVariables.furthestmembranepos(1) = trackingParameters.vesicle_datatip_coordinates(1,2);
% trackingVariables.furthestmembranepos(2) = trackingParameters.vesicle_datatip_coordinates(1,1);
% trackingVariables.center(1) = trackingParameters.vesicle_datatip_coordinates(2,2);
% trackingVariables.center(2) = trackingParameters.vesicle_datatip_coordinates(2,1);

trackingVariables.contourclosedindex = zeros(programParameters.contourNr,1); % this will hold an index of whether a membrane was closed or not

trackingVariables.outside = 1;	% this switch serves to check how many times the
                                % tracking algorithm wanderd out of bounds
                            
trackingVariables.abort = false;	% this variable serves to break of the detection 
                                    % loop if the user wishes to do so; it works together 
                                    % with the cancel pushbutton of the status
                                    % dialog
                            
% Control variables
trackingVariables.method = zeros(trackingParameters.maxnrofpoints,1); %variable to check what trackingVariables.method was used at what iteration
trackingVariables.methodinternaldirection = zeros(trackingParameters.maxnrofpoints,1); % control variable to see which direction was used inside a direction methods
trackingVariables.deltaSxy = zeros(trackingParameters.maxnrofpoints,1);
trackingVariables.deltaSxw = zeros(trackingParameters.maxnrofpoints,1);
trackingVariables.deltaSxv = zeros(trackingParameters.maxnrofpoints,1);

trackingVariables.detectedpoints = zeros(programParameters.contourNr,1); % array that will contain the detected points per contour

trackingVariables.outsidecounter = 0; % counter to count how many times the algorithm wandered out of
                    % the image and was stopped because of that

trackingVariables.contourclosedcounter = 0; % counter to count the number of time the contour was closed

% xbar,ybar,vbar,wbar: postions of membrane for calculation of next membrane point (xtild,ytild)
% Sx1,Sx2,Sxbar,Sy1,Sy2,Sybar: slopes of membranes

% Counter/Index Variables:
% index1: index for looping through the images
% index2: index for looping through the detected pixels

trackingVariables.xDirectionChangeCounter = 0; % these are counter so that a new direction-change can only occur after a certain amount of iterations
trackingVariables.yDirectionChangeCounter = 0; % given by the parameter 'trackingParameters.directiondetectionstart'
trackingVariables.vDirectionChangeCounter = 0;
trackingVariables.wDirectionChangeCounter = 0;

trackingVariables.centercounter = 0;  % this is a counter to be able to calculate a new contour
                                      % trackingVariables.center after the number of steps given by the parameter 
                                      % 'trackingParameters.newcentersteps'
                    

% variables indicating the x- and y-directions to the direction of the last breakofpoint
trackingVariables.xdirection = 1;
trackingVariables.ydirection = 1;
trackingVariables.vdirection = 1;
trackingVariables.wdirection = 1;

% variable used in function to signal break-off of detection loop
trackingVariables.skipImage = 0;