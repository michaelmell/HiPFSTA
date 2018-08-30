classdef flickeringDataClass < handle
    properties
        closedContours;
        closedContourIndexes;
        goodContourIndexes;
        badContourIndexes;
        deletedContours;
        parameterStruct;
        fourierseries;
        labelString;
        idString;
        contour = struct([]);
        polarAngles;
        polarRadii;
        cartesianX;
        cartesianY;
        validContours;
    end

    properties (Access=private)
    end
    
    methods (Access=private)
        function obj = calculateReferenceCenters( obj )
            switch obj.parameterStruct.centerMethod
                case 'forEachContour'
                    for index = 1:length(obj.contour)
                        obj.contour(index).referenceCenter = obj.contour(index).barioCenter;
                    end
                case 'meanCenter'
%                     centers = obj.getBarioCenters;
                    centers = obj.getBarioCentersInternal;
                    meanCenter = nanmean(centers,1);
                    for index = 1:length(obj.contour)
                        obj.contour(index).referenceCenter = meanCenter;
                    end
            end
        end
    end
    
    methods % setter and getter methods
%         function set.polarAngles( obj, value )
%             obj.polarAngles = value;
%         end
%         function result = get.polarAngles(obj)
%             result = obj.polarAngles;
%             if ~isempty(obj.goodContourIndexes)
%                 result = result(:,obj.goodContourIndexes);
%             end
%         end
%         
%         function set.polarRadii( obj, value )
%             obj.polarRadii = value;
%         end
%         function result = get.polarRadii(obj)
%             result = obj.polarAngles;
%             if ~isempty(obj.goodContourIndexes)
%                 result = result(:,obj.goodContourIndexes);
%             end
%         end
%         
%         function set.cartesianX( obj, value )
%             obj.cartesianX = value;
%         end
%         function result = get.cartesianX(obj)
%             result = obj.cartesianX;
%             if ~isempty(obj.goodContourIndexes)
%                 result = result(:,obj.goodContourIndexes);
%             end
%         end
%         
%         function set.cartesianY( obj, value )
%             obj.cartesianY = value;
%         end
%         function result = get.cartesianY(obj)
%             result = obj.cartesianY;
%             if ~isempty(obj.goodContourIndexes)
%                 result = result(:,obj.goodContourIndexes);
%             end
%         end
        function obj = setFps(obj,fps)
            obj.parameterStruct.fps = fps;
        end
        function fps = getFps(obj)
            fps = obj.parameterStruct.fps;
        end
        
        function result = getMeanRadius(obj)
            result = mean(obj.getRadiusSeries);
        end
        
        function timebase = getTimebase(obj)
            timebase = (1:size(obj.cartesianX,2))/obj.parameterStruct.fps;
        end

        function obj = setIdString( obj, string )
            obj.idString = string;
        end
        
        function result = getIdString(obj)
            result = obj.idString;
        end
        
        function obj = setLabelString( obj, string )
            obj.labelString = string;
        end
        
        function result = getLabelString(obj)
            result = obj.labelString;
        end
        
        function obj = setResolution( obj, resolution ) % the optical resultion/pixel at which the movie was recorded
            obj.parameterStruct.resolution = resolution;
        end
        
        function result = getResolution( obj )
            result = obj.parameterStruct.resolution;
        end
        
        function obj = setMaxLengthDeviation( obj, maxlengthdeviation) % maximum allowed deviation (in %) of the contour-length from its mean value
            obj.parameterStruct.maxlengthdeviation = maxlengthdeviation;
        end
        
        function result = getMaxLengthDeviation( obj )
            result = obj.parameterStruct.maxlengthdeviation;
        end
        
        
        function [minContourLength,contourIndex] = getMinContourLength( obj )
            nrOfContours = obj.getNrOfContours;
            result = nan(nrOfContours,1);
%             contourIndex = [];
            for index = 1:nrOfContours
                nonNaNindices = ~isnan(obj.cartesianX(:,index));
%                 result = length(obj.cartesianX(nonNaNindices,index));
                result(index) = length(obj.cartesianX(nonNaNindices,index));
            end
            
%             result = min(result);
            [minContourLength,contourIndex] = min(result);
            if length(contourIndex) > 1
                minContourLength = minContourLength(1);
                contourIndex = contourIndex(1);
            end
        end

        function meanContourLength = getMeanContourLength( obj )
            nrOfContours = obj.getNrOfContours;
            result = nan(nrOfContours,1);
%             contourIndex = [];
            for index = 1:nrOfContours
                nonNaNindices = ~isnan(obj.cartesianX(:,index));
%                 result = length(obj.cartesianX(nonNaNindices,index));
                result(index) = length(obj.cartesianX(nonNaNindices,index));
            end
            
%             result = min(result);
%             [meanContourLength,contourIndex] = min(result);
            meanContourLength = mean(result);
        end
        
        function [polarAngles,polarRadii] = getPolarAtIndex( obj, coordIndex )
            columnIndex = (1:length(coordIndex));
            linearindex = sub2ind(size(obj.polarAngles), coordIndex, columnIndex);
            polarAngles = obj.polarAngles(:);
            polarAngles = polarAngles(linearindex);
            polarRadii = obj.polarRadii(:);
            polarRadii = polarRadii(linearindex);
        end
        
        function [cartesianX,cartesianY] = getCartesianAtIndex( obj, coordIndex )
            columnIndex = (1:length(coordIndex));
            linearindex = sub2ind(size(obj.polarAngles), coordIndex, columnIndex);
            cartesianX = obj.cartesianX(:);
            cartesianX = cartesianX(linearindex);
            cartesianY = obj.cartesianY(:);
            cartesianY = cartesianY(linearindex);
        end
        
%         function result = getAngleSeries( obj, radianAngle )
        function [cartesianX,cartesianY,coordIndex] = getCartesianSeriesAtAngle( obj, radianAngle )
            [value,coordIndex] = min( abs( obj.polarAngles - radianAngle ) );
            columnIndex = (1:length(coordIndex));
            linearindex = sub2ind(size(obj.polarAngles), coordIndex, columnIndex);
            cartesianX = obj.cartesianX(:);
            cartesianX = cartesianX(linearindex);
            cartesianY = obj.cartesianY(:);
            cartesianY = cartesianY(linearindex);
        end
        
        function [result,coordIndex] = getSeriesAtAngle( obj, radianAngle )
            [value,coordIndex] = min( abs( obj.polarAngles - radianAngle ) );
            columnIndex = (1:length(coordIndex));
            linearindex = sub2ind(size(obj.polarAngles), coordIndex, columnIndex);
            result = obj.polarRadii(:);
            result = result(linearindex);
        end
        
        function [result,coordIndex] = getSeriesAtAngleInPixels( obj, radianAngle )
            [result,coordIndex] = obj.getSeriesAtAngle( radianAngle );
            result = result /obj.parameterStruct.resolution;
        end
        
        function [angle,radius,coordIndex] = getContourByAngle( obj, index, angleRange )
            if size(angleRange,1)>size(angleRange,2)
                angleRange = transpose(angleRange); % angleRange needs to be row-vector
            end
            polarCoords = obj.getContourPolar( index );
            angles = polarCoords(:,2);
            radii = polarCoords(:,1);
            radius = nan(length(angleRange),1);
            angle = nan(length(angleRange),1);
            coordIndex = nan(length(angleRange),1);
            for index = 1:length(angleRange)
                [value,coordIndex(index)] = min( abs( angles - angleRange(index) ) );
                radius(index) = radii(coordIndex(index));
                angle(index) = angles(coordIndex(index));
            end
        end
        
        function [angle,radius,coordIndex] = getContourByAngleInPixels( obj, index, angleRange )
            [angle,radius,coordIndex] = obj.getContourByAngle( index, angleRange );
            radius = radius/obj.parameterStruct.resolution;
        end
        
        function obj = setReferenceCenterMethod( obj, centerMethod )
            % sets how to determine the center of each contour for the
            % conversion from cartesian to polar coordinates;
            % possible values:
            % 'forEachContour': use respective center of every contour
            % 'meanCenter': use mean value of all centers
            obj.parameterStruct.centerMethod = centerMethod;
            obj.calculateReferenceCenters();
        end
        
        function obj = setNrOfModes( obj, nmax )
            obj.parameterStruct.nmax = nmax;
        end
        
        function result = getReferenceCenters( obj )
            % note these are the centers used for converting the contour to
            % polar coordinates
            result = nan(length(obj.contour),2);
            for index = 1:length(obj.contour)
                result(index,:) = obj.contour(index).referenceCenter;
            end
        end
        
        function result = getBarioCenters( obj )
            result = nan(obj.getNrOfContours,2);
            for index1 = 1:length(obj.contour)
                index = obj.mapIndexToGoodContours( index1 );
                result(index,:) = obj.contour(index).barioCenter;
            end
        end
        
        function result = getBarioCentersInternal( obj )
            nrOfContours = size(obj.cartesianX,2);
            result = nan(nrOfContours,2);
%             result = nan(obj.getNrOfContours,2);
%             for index1 = 1:length(obj.contour)

            goodContourIndexesSize = size(obj.goodContourIndexes);
            if goodContourIndexesSize(1) > 1
                goodContourIndexesTMP = transpose(obj.goodContourIndexes);
            else
                goodContourIndexesTMP = obj.goodContourIndexes;
            end
            
%             for contourIndex = obj.goodContourIndexes
            for index = goodContourIndexesTMP
%             for index = 1:nrOfContours
%                 index = obj.mapIndexToGoodContours( index1 );
                result(index,:) = obj.contour(index).barioCenter;
            end
        end
        
%         function result = getOldRadiusSeries( obj )
%             result = nan(size(obj.contour));
%             for index = 1:length(obj.contour)
%                 result(index) = mean( obj.contour(index).oldPolar(:,1) );
%             end
%         end
        
        function result = getRadiusSeries( obj )
            result = nan(obj.getNrOfContours,1);
%             for index = 1:obj.getNrOfContours
            if isfield(obj.contour(1),'radius')
                for index = obj.goodContourIndexes
    %                 index = obj.mapIndexToGoodContours( index1 );
                    result(index) = obj.contour(index).radius;
                end
            else
                for contourNr = obj.goodContourIndexes
    %                 disp(['Fourier Transform Contour ',num2str(contourNr),' of ', num2str(nrOfContours)]);

                    [angles,radii] = obj.getContourPolarPhaseCorrectedInternal(contourNr);

    %                 [circumference,ds] = obj.calculateContourCircumference(contourNr);
    %                 radius = (1/(4*pi)) * circumference * obj.parameterStruct.resolution;
    % %                 r0(j) = (1/(4*pi)) * tmp2;

                    mempos = nan(length(angles),2);
                    mempos(:,1) = radii;
                    mempos(:,2) = angles;

                    result(contourNr) = obj.calculateMode0radius(mempos);
                    obj.contour(contourNr).radius = result(contourNr);
                end
            end
        end
        
        function result = getRadiusSeriesAtAngle( obj, polarAngle )
            result = obj.fourierseries(modeNr,:);
%             radii = datasets{index}.polarRadius(:,frameNr);
%             angles = datasets{index}.polarAngle(:,frameNr);
        end
        
        function result = getRadiusTimeTraceAtAngle( obj, polarAngle )
            result = obj.fourierseries(modeNr,:);
%             radii = datasets{index}.polarRadius(:,frameNr);
%             angles = datasets{index}.polarAngle(:,frameNr);
        end
        
        function timeTrace = getRadiusTimeTraceAtIndex( obj, coordIndex )
            radii = obj.polarRadii(coordIndex,:);
%             angles = obj.polarAngle(coordIndex,:);
            timeTrace = timeTraceClass();
            set(timeTrace,'data',radii);
            if isfield(obj.parameterStruct,'fps')
                set(timeTrace,'timeStepLength',1/obj.parameterStruct.fps);
            else
                error('flickeringDataClass:popertyCheck','The FPS information for the flickering dataset has not been set. Set it using the ''setFps'' method.');
            end
        end
        
        function result = getModeSeries( obj, modeNr )
            result = obj.fourierseries(modeNr,:);
        end
        
        function timeTrace = getModeTimeTrace( obj, modeNr )
            modeSeries = obj.getModeSeries( modeNr );
            timeTrace = timeTraceClass();
            set(timeTrace,'data',modeSeries);
            if isfield(obj.parameterStruct,'fps')
                set(timeTrace,'timeStepLength',1/obj.parameterStruct.fps);
            else
                error('flickeringDataClass:popertyCheck','The FPS information for the flickering dataset has not been set. Set it using the ''setFps'' method.');
            end
            meanRadius = obj.getMeanRadius();
            set(timeTrace,'wavenumber',modeNr/meanRadius);
        end
        
        function result = getCircumferenceSeries( obj )
try
            result = nan(obj.getNrOfContours,1);
            for index1 = 1:obj.getNrOfContours
                index = obj.mapIndexToGoodContours( index1 );
                result(index) = mean( obj.contour(index).circumference );
            end
catch
    keyboard
end
        end
        
        function result = getSumDsMeanSeries( obj )
            result = nan(obj.getNrOfContours);
            for index1 = 1:length(obj.contour)
                index = obj.mapIndexToGoodContours( index1 );
                result(index) = mean( obj.contour(index).sumds );
            end
        end
        
        function result = getContourCartesianInMeters( obj, index )
            if index <= obj.getNrOfContours
                index = obj.mapIndexToGoodContours( index );
                result = obj.contour(index).cartesian * obj.parameterStruct.resolution;
            else
                result = [];
            end
        end
        
        function index = mapIndexToGoodContours( obj, index )
            index = obj.goodContourIndexes( index );
        end
        
        function result = getNrOfContours( obj )
            if ~isempty(obj.goodContourIndexes)
                result = length( obj.goodContourIndexes );
            else
                result = size(obj.polarAngles,2);
            end
        end
        
        function result = getContourCartesian( obj, index )
%             if index <= obj.getNrOfContours
            if index <= size(obj.cartesianX,2) % IMPORTANT: this should not be changed to using getNrOfContours
                try
                    nonNaNindices = ~isnan(obj.cartesianX(:,index));
    %                 result = [obj.cartesianX(nonNaNindices,index), obj.cartesianY(nonNaNindices,index)];
                    result = [obj.cartesianY(nonNaNindices,index),obj.cartesianX(nonNaNindices,index)];
                catch
                    keyboard
                end
            else
                error('myApp:argChk', ['Contour index=',num2str(index), ...
                                       ' out of bounds; Nr. of contours is ',num2str(obj.getNrOfContours)'.'])
%                 disp();
                result = [];
            end
        end
%         function result = getContourCartesian( obj, index )
%             if index < length(obj.contour)
%                 result = obj.contour(index).cartesian;
%             else
%                 result = [];
%             end
%         end

        function [angles,radii] = getContourPolarPhaseCorrected( obj, index )
            % Decscription:
            % This method does the same as 'getContourPolar', but shifts
            % the array containing the coordinates, so that all contours
            % start approximately with the same angle phi=0.
%             polarCoords = obj.getContourPolar( index );
            polarCoords = obj.getGoodContourPolar( index );
            radii = polarCoords(:,1);
            angles = polarCoords(:,2);
%             [value,coordIndex] = min( abs( angles ) );
%             [value,coordIndex] = min( angles + pi ); % shift matrix so that coordinates start with -pi
%             [value,coordIndex] = min( angles ); % shift matrix so that coordinates start with -pi
%             if length(coordIndex) > 1
%                 coordIndex = coordIndex(1);
%             end
%             radii = [radii(coordIndex:end);radii(1:coordIndex-1)];
%             angles = [angles(coordIndex:end);angles(1:coordIndex-1)];
            [angles,sortIndices] = sort(angles);
            radii = radii(sortIndices);
        end
        
        function [angles,radii] = getContourPolarPhaseCorrectedInternal( obj, index )
            % Decscription:
            % This method does the same as 'getContourPolar', but shifts
            % the array containing the coordinates, so that all contours
            % start approximately with the same angle phi=0.
%             polarCoords = obj.getContourPolar( index );
            nonNaNindices = ~isnan(obj.polarRadii(:,index));
            polarCoords = [obj.polarRadii(nonNaNindices,index), obj.polarAngles(nonNaNindices,index)];
            radii = polarCoords(:,1);
            angles = polarCoords(:,2);
%             [value,coordIndex] = min( abs( angles ) );
%             [value,coordIndex] = min( angles + pi ); % shift matrix so that coordinates start with -pi
%             [value,coordIndex] = min( angles ); % shift matrix so that coordinates start with -pi
% keyboard
%             if length(coordIndex) > 1
%                 coordIndex = coordIndex(1);
%             end
%             radii = [radii(coordIndex:end);radii(1:coordIndex-1)];
%             angles = [angles(coordIndex:end);angles(1:coordIndex-1)];
            
            [angles,sortIndices] = sort(angles);
            radii = radii(sortIndices);
        end
        
        function result = getContourPolar( obj, index )
            if index <= obj.getNrOfContours
                nonNaNindices = ~isnan(obj.polarRadii(:,index));
                result = [obj.polarRadii(nonNaNindices,index), obj.polarAngles(nonNaNindices,index)];
            else
                result = [];
            end
        end
        
        function result = getGoodContourPolar( obj, index )
% keyboard
            if index <= obj.getNrOfContours
%                 frameNrCorrected = dataset{datasetIndex}.mapIndexToGoodContours(frameNr);
                goodIndex = obj.mapIndexToGoodContours(index);
                nonNaNindices = ~isnan(obj.polarRadii(:,goodIndex));
                result = [obj.polarRadii(nonNaNindices,goodIndex), obj.polarAngles(nonNaNindices,goodIndex)];
            else
                result = [];
            end
        end
%         function result = getContourPolar( obj, index )
%             if index < length(obj.contour)
%                 result = obj.contour(index).polar;
%             else
%                 result = [];
%             end
%         end
    end
    
    methods
        function obj = flickeringDataClass( varargin )
          if(obj.isOctave())
            pkg load statistics;
          end
        end
        
        function savedStructure = saveobj(obj) % for details see: http://www.mathworks.es/es/help/matlab/ref/saveobj.html
            fields = fieldnames(obj);
            for index = 1:length(fields)
                savedStructure.(fields{index}) = obj.(fields{index});
            end
        end
    end
    
    methods (Static = true)
        function obj = loadobj(savedStructure) % for details see: http://www.mathworks.es/es/help/matlab/ref/loadobj.html
            obj = flickeringDataClass();
            fields = fieldnames(savedStructure);
            for index = 1:length(fields)
                obj.(fields{index}) = savedStructure.(fields{index});
            end
            if isfield(obj.contour,'cartesian')
                obj = writePolarCoordinatesToArray(obj);
                obj = writeCartesianCoordinatesToArray(obj);
            end
        end

        %%
        %% Return: true if the environment is Octave.
        %%
        function retval = isOctave
          persistent cacheval;  % speeds up repeated calls

          if isempty (cacheval)
            cacheval = (exist ("OCTAVE_VERSION", "builtin") > 0);
          end

          retval = cacheval;
        end
    end
    
    methods 
        function obj = writePolarCoordinatesToArray(obj)
            polarCoords = arrayfun( @(x) x.polar , obj.contour, 'uniformoutput', false );
            profileLengths = cellfun('length',polarCoords); % same for polar and cartesian coordinates
            obj.polarAngles = nan(max(profileLengths),length(polarCoords));
            obj.polarRadii = nan(max(profileLengths),length(polarCoords));
            angles = cellfun(@(x) x(:,2),polarCoords, 'uniformoutput', false);
            radii = cellfun(@(x) x(:,1),polarCoords, 'uniformoutput', false);

            for index = 1:length(polarCoords)
                obj.polarAngles(1:length(angles{index}),index) = angles{index};
                obj.polarRadii(1:length(angles{index}),index) = radii{index};
            end
        end
        
        function obj = writePolarCoordinatesToArrayNew(obj,polarCoords)
%             polarCoords = arrayfun( @(x) x.polar , obj.contour, 'uniformoutput', false );
            profileLengths = cellfun('length',polarCoords); % same for polar and cartesian coordinates
% try         
    %%
            badContourIndexesTMP = transpose(find(profileLengths==0));
            if ~isempty(badContourIndexesTMP)
                for badContourIndexesTMP = transpose(find(profileLengths==0))
                    polarCoords{badContourIndexesTMP} = nan(max(profileLengths),2);
                end
            end
    %%
% catch 
%     keyboard
% end
            obj.polarAngles = nan(max(profileLengths),size(polarCoords,1));
            obj.polarRadii = nan(max(profileLengths),size(polarCoords,1));
            angles = cellfun(@(x) x(:,2),polarCoords, 'uniformoutput', false);
            radii = cellfun(@(x) x(:,1),polarCoords, 'uniformoutput', false);
            
            for index = 1:length(polarCoords)
                obj.polarAngles(1:length(angles{index}),index) = angles{index};
                obj.polarRadii(1:length(angles{index}),index) = radii{index};
            end
            
        end
        
        function obj = writeCartesianCoordinatesToArray(obj)
            cartesianCoords = arrayfun( @(x) x.cartesian , obj.contour, 'uniformoutput', false );
            profileLengths = cellfun('length',cartesianCoords); % same for polar and cartesian coordinates
            obj.cartesianX = nan(max(profileLengths),length(cartesianCoords));
            obj.cartesianY = nan(max(profileLengths),length(cartesianCoords));
            x = cellfun(@(x) x(:,1),cartesianCoords, 'uniformoutput', false);
            y = cellfun(@(x) x(:,2),cartesianCoords, 'uniformoutput', false);
            
            for index = 1:length(cartesianCoords)
                obj.cartesianX(1:length(x{index}),index) = x{index};
                obj.cartesianY(1:length(x{index}),index) = y{index};
            end
        end
        
        function obj = writeCartesianCoordinatesToArrayOnNewRead(obj,contourCoordinates)
%             cartesianCoords = arrayfun( @(x) x.cartesian , obj.contour, 'uniformoutput', false );
            cartesianCoords = contourCoordinates;
            profileLengths = cellfun('length',cartesianCoords); % same for polar and cartesian coordinates
            obj.cartesianX = nan(max(profileLengths),length(cartesianCoords));
            obj.cartesianY = nan(max(profileLengths),length(cartesianCoords));
            x = cellfun(@(x) x(:,1),cartesianCoords, 'uniformoutput', false);
            y = cellfun(@(x) x(:,2),cartesianCoords, 'uniformoutput', false);
            
            for index = 1:length(cartesianCoords)
                obj.cartesianX(1:length(x{index}),index) = x{index};
                obj.cartesianY(1:length(x{index}),index) = y{index};
            end
        end
    end
    
    methods % setters and getters
%         function val = get.frame(obj)
%             val = obj.frame;
%         end
    end
    
    methods
        function obj = loadContoursOld( obj, path )
            obj.parameterStruct.datasetLoadPath = path;
            fileList = dir([path,'/contours/contour*.txt']);
%             obj.deletedContours = load([path,'/results/deletedcontours.txt']);
%             if isempty(obj.deletedContours)
%                 obj.deletedContours = nan;
%             end
            obj.closedContours = load([path,'/results/contourclosedindex.txt']);
            
            nrOfContours = length(fileList);
            contourCoordinates = cell(nrOfContours,1);
            
            for fileCounter = 1:nrOfContours
                disp(['Loading profile ',num2str(fileCounter),' of ', num2str(nrOfContours)]);
                contourCoordinates{fileCounter} = load([path,'/contours/contour',num2str(fileCounter),'.txt']);
            end
            emptyCells = cellfun(@isempty,contourCoordinates);
            contourCoordinates(emptyCells) = [];
            obj = writeCartesianCoordinatesToArrayOnNewRead(obj,contourCoordinates);

            obj.closedContourIndexes = find(obj.closedContours==1);
            obj.closedContourIndexes = transpose(obj.closedContourIndexes);
        end
%         function obj = loadContoursOld( obj, path )
%             obj.parameterStruct.datasetLoadPath = path;
%             fileList = dir([path,'/contours/contour*.txt']);
%             obj.deletedContours = load([path,'/results/deletedcontours.txt']);
%             if isempty(obj.deletedContours)
%                 obj.deletedContours = nan;
%             end
%             obj.closedContours = load([path,'/results/contourclosedindex.txt']);
%             
%             nrOfContours = length(fileList);
%             fileCounter = 1;
%             index = 1;
%             contourCoordinates = cell(nrOfContours,1);
%             
%             while (1==1)
%                 disp(['Loading profile ',num2str(index),' of ', num2str(nrOfContours)]);
%                 if obj.closedContours(fileCounter) == 1 && ... % first check if contour should be omitted
%                    ~any( obj.deletedContours == fileCounter ) && ... % second check if contour should be omitted
%                    exist([path,'/contours/contour',num2str(fileCounter),'.txt'],'file')
%                
% %                     cartesian = load([path,'/contours/contour',num2str(fileCounter),'.txt']);
% %                     obj.contour(index).cartesian = load([path,'/contours/contour',num2str(fileCounter),'.txt']);
%                     contourCoordinates{index} = load([path,'/contours/contour',num2str(fileCounter),'.txt']);
% %                     obj.contour(index).oldPolar = load([path,'/contours/polarcontour',num2str(fileCounter),'.txt']);
% %                     obj.frame(index).contour = load([path,'/contours/contour',num2str(fileCounter),'.txt']);
%                     index = index + 1;
%                 end
%                 if fileCounter == nrOfContours;
%                     break;
%                 end
%                 fileCounter = fileCounter + 1;
%             end
%             emptyCells = cellfun(@isempty,contourCoordinates);
%             contourCoordinates(emptyCells) = [];
%             obj = writeCartesianCoordinatesToArrayOnNewRead(obj,contourCoordinates);
% 
%             obj.closedContourIndexes = find(obj.closedContours==1);
%         end

        function obj = loadContours( obj, path )
            obj.parameterStruct.datasetLoadPath = path;

%             fileList = dir([path,'/contours/contour*.txt']);
%             obj.deletedContours = load([path,'/results/deletedcontours.txt']);
%             if isempty(obj.deletedContours)
%                 obj.deletedContours = nan;
%             end
            obj.closedContours = load([path,'/contourclosedindex.txt']);
            tmp = load([path,'/contourCoordinates.mat']);
            contourCoordinates = tmp.contourCoordinates;
            emptyCells = cellfun(@isempty,contourCoordinates);
            contourCoordinates(emptyCells) = [];
            obj = writeCartesianCoordinatesToArrayOnNewRead(obj,contourCoordinates);
            obj.closedContourIndexes = find(obj.closedContours==1);
            tmp = load([path,'/parameterStruct.mat']);
            obj.parameterStruct = tmp.parameterStruct;
        end

        function obj = loadImageRotateSettings_v000( obj, path )
            tmp = load([path,'/parameterStruct.mat']);
            obj.parameterStruct = tmp.parameterStruct;
        end
        
        function obj = loadImageRotationContours_v000( obj, path )
            obj.parameterStruct.datasetLoadPath = path;
            
            % load tracked polar coordinates
            tmp = load([path,'/polarCoordinates.mat']);
            polarCoordinates = tmp.polarCoordinates;
            newPolarCoordinates = cell(size(polarCoordinates));
            
            for index = 1:length(polarCoordinates)
%                 angle = polarCoordinates{index}(:,1) - pi; % convert from 0:2*pi to -pi:pi
                angle = polarCoordinates{index}(:,1);
                radius =  polarCoordinates{index}(:,2) * obj.parameterStruct.resolution;
%                 polarContour = [radius,angle];
                angle = angle-pi/2;
%                 angle = angle; % rotate so that it coincides with the image
                radius = [radius(angle>pi);radius(angle<=pi)]; % move range to -pi:pi
                angle = [angle(angle>pi)-2*pi;angle(angle<=pi)]; % move range to -pi:pi
                polarContour = [radius,angle];
                newPolarCoordinates{index} = polarContour;
            end
            obj = obj.writePolarCoordinatesToArrayNew(newPolarCoordinates);
            obj.goodContourIndexes = (1:length(polarCoordinates));
            obj.calculateCartesianCoordinates();
        end

        function obj = loadImageInterpolationSettings_v002( obj, path )
            tmp = load([path,'/parameterStruct.mat']);
            obj.parameterStruct = tmp.parameterStruct;
        end
        
        function obj = loadImageInterpolationContours_v002( obj, path )
            obj.parameterStruct.datasetLoadPath = path;

            % load tracked polar coordinates
            tmp = load([path,'/polarCoordinates.mat']);
            polarCoordinates = tmp.polarCoordinates;
            newPolarCoordinates = cell(size(polarCoordinates));
            
            counter = 1;
            for index = 1:length(polarCoordinates)
%                 angle = polarCoordinates{index}(:,1) - pi; % convert from 0:2*pi to -pi:pi
                if ~isempty(polarCoordinates{index})
                    angle = polarCoordinates{index}(:,1);
                    radius =  polarCoordinates{index}(:,2) * obj.parameterStruct.resolution;
    %                 polarContour = [radius,angle];
                    angle = angle-pi/2;
    %                 angle = angle; % rotate so that it coincides with the image
                    radius = [radius(angle>pi);radius(angle<=pi)]; % move range to -pi:pi
                    angle = [angle(angle>pi)-2*pi;angle(angle<=pi)]; % move range to -pi:pi
                    polarContour = [radius,angle];
                    newPolarCoordinates{index} = polarContour;
                    obj.goodContourIndexes(counter) = index;
                else
                    newPolarCoordinates{index} = [NaN,NaN];
                    obj.badContourIndexes(counter) = index;
                end
                counter = counter + 1;
            end
            obj = obj.writePolarCoordinatesToArrayNew(newPolarCoordinates);
%             obj.goodContourIndexes = (1:length(polarCoordinates));
            obj.calculateCartesianCoordinates();
        end

        function obj = calculateCartesianCoordinates(obj)
            % convert tracked polar coordinates to cartesian coordinates
            resolution = obj.getResolution;
            trackingCenter = obj.parameterStruct.roiCenter;
            cartesianCoordinates = cell(1,size(obj.polarAngles,2));
            
            for index = 1:size(obj.polarAngles,2)
                polarCoords = obj.getContourPolar( index );
                angle = polarCoords(:,2);
                radius = polarCoords(:,1);
                radius = radius/resolution;
                [x,y] = pol2cart(angle,radius);
                cartesianCoordinates{index} = [y+trackingCenter(1),x+trackingCenter(2)];
            end
            
            obj = writeCartesianCoordinatesToArrayOnNewRead(obj,cartesianCoordinates);
        end
        
        function obj = loadPythonTrackingContours_v000( obj, path )
            tmp = load([path,'/contourCoordinatesX.mat']);
            contourCoordinatesX = tmp.contourCoordinatesX;
            obj.cartesianX = contourCoordinatesX;
            tmp = load([path,'/contourCoordinatesY.mat']);
            contourCoordinatesY = tmp.contourCoordinatesY;
            obj.cartesianY = contourCoordinatesY;
            
            obj.goodContourIndexes = (1:size(obj.cartesianX,2));
            obj.badContourIndexes = [];
            
            indexCounter = 1;
            for contourIndex = length(obj.goodContourIndexes):-1:1
                if any(contourCoordinatesX(:,contourIndex)==0) || any(isnan(contourCoordinatesX(:,contourIndex)))
                    obj.goodContourIndexes(contourIndex) = [];
                    obj.badContourIndexes(indexCounter) = contourIndex;
                    indexCounter = indexCounter + 1;
                end
%                 contourCoordinatesX(:,1801)
            end
            
            obj.badContourIndexes = sort(obj.badContourIndexes);
            obj.goodContourIndexes = sort(obj.goodContourIndexes);
            obj.parameterStruct.mincontourlength = 0;
            obj.parameterStruct.maxangle = 180;
            obj.parameterStruct.maxlengthdeviation = 5;
        end
        
        function obj = testContours(obj)
%             counter = 1;
            obj.goodContourIndexes(obj.goodContourIndexes==0) = [];

            counter = 1;
            if isempty(obj.closedContourIndexes)
                obj.closedContourIndexes = obj.goodContourIndexes;
            end
            
            tmp1 = [];
            for contourIndex = obj.closedContourIndexes
                if precheck1(obj.parameterStruct,obj.getContourCartesian(contourIndex));
%                     obj.goodContourIndexes(counter) = contourIndex;
                    tmp1(counter) = contourIndex;
                    counter = counter + 1;
                end
            end
            obj.goodContourIndexes = tmp1;
            
            tmp = obj.goodContourIndexes;
            counter = 1;
            for contourIndex = tmp
                if ~method1(obj.parameterStruct,obj.getContourCartesian(contourIndex))
                    obj.goodContourIndexes(counter) = [];
                end
                counter = counter + 1;
            end

            obj.goodContourIndexes = method2(obj);
            
            function contourIsGood = precheck1(parameterStruct,contour)
                % Precheck:
                % this checks to see if the vector has at least
                % [parameterStruct.mincontourlength] points inside it
                contourIsGood = true;
                if length(contour) < parameterStruct.mincontourlength
                    contourIsGood = false;
                end
            end
            
            function contourIsGood = method1(parameterStruct,mempos)
                % Method 1:
                % check that the closing angle is less than a given angle
                contourIsGood = true;

                a = mempos(10,:) - mempos(1,:);
                b = mempos(length(mempos),:) - mempos(length(mempos) - 9,:);
                
                % calculate angle:
                alpha = acos((a*transpose(b))/(norm(a)*norm(b)));

                if alpha > parameterStruct.maxangle
                    contourIsGood = false;
                end
            end
            
            function goodContoursIndexes = method2(obj)
                obj.calculateCircumference();

                circumferences = obj.getCircumferenceSeries;
%                 circumferences = circumferences(obj.goodContourIndexes);
                meanCircumference = mean(circumferences(obj.goodContourIndexes));
                onepercentofmeancircumference = meanCircumference/100;
%                 contourIsGood = true;
%                 if abs(L(j) - meancircumference)>onepercentofmeancircumference*parameterStruct.maxlengthdeviation
%                     contourIsGood = false;
%                 end
                goodContourMask = abs(circumferences - meanCircumference) < onepercentofmeancircumference*obj.parameterStruct.maxlengthdeviation;
                goodContoursIndexes = (find( goodContourMask == 1 ));
            end
        end

        function obj = calculateCircumference(obj)
%             nrOfContour = size(obj.cartesianX,2);
            nrOfContour = length(obj.goodContourIndexes);
            
            goodContourIndexesSize = size(obj.goodContourIndexes);
            if goodContourIndexesSize(1) > 1
                goodContourIndexesTMP = transpose(obj.goodContourIndexes);
            else
                goodContourIndexesTMP = obj.goodContourIndexes;
            end
            
%             for contourIndex = obj.goodContourIndexes
            for contourIndex = goodContourIndexesTMP
                disp(['Calculating circumference ',num2str(contourIndex),' of ', num2str(nrOfContour)]);

                [circumference,ds] = obj.calculateContourCircumference(contourIndex);

                obj.contour(contourIndex).circumference = circumference;
                obj.contour(contourIndex).ds = ds;
            end
        end
        
        function [circumference,ds] = calculateContourCircumference(obj,contourIndex)
                circumference = 0; % length of circumference
                cartesianContour = obj.getContourCartesian(contourIndex);
                memposlength = length(cartesianContour); % length of the vector mempos
                ds = zeros(memposlength,1); % vector containing the distances between neighbooring membrane positions
                
                % calculate length of circumference
                for i = 1:memposlength - 1
                    ds(i + 1) = sqrt((cartesianContour(i + 1,1) - cartesianContour(i,1))^2 ...
                                  + (cartesianContour(i + 1,2) - cartesianContour(i,2))^2 ...
                                  );
                    circumference = circumference + ds(i);
                end
                
                % to get the last part between the end and starting point of tracked contour
                ds(1) = sqrt((cartesianContour(1,1) - cartesianContour(i + 1,1))^2 ...
                           + (cartesianContour(1,2) - cartesianContour(i + 1,2))^2 ...
                           );
                circumference = circumference + ds(1);
        end
        
        function obj = calculateBarioCenter(obj)
%             nrOfContour = obj.getNrOfContours;
%             nrOfContour = size(obj.cartesianX,2);
%             for index = 1:nrOfContour
            goodContourIndexesSize = size(obj.goodContourIndexes);
            if goodContourIndexesSize(1)>1
                goodContourIndexesTMP = transpose(obj.goodContourIndexes);
            else
                goodContourIndexesTMP = obj.goodContourIndexes;
            end
            
            for index = goodContourIndexesTMP
% try
%             for index = 1:length(obj.contour)
%                 disp(['Calculating circumference ',num2str(index),' of ', num2str(length(obj.contour))]);
                mempos = obj.getContourCartesian(index);
%                 mempos = obj.contour(index).cartesian;
                
                ds = obj.contour(index).ds;
                L = obj.contour(index).circumference;
                memposlength = length(mempos); % length of the vector mempos
                sumds = zeros(memposlength,1); % vector containing sum between ds(i) and ds(i + 1)
%                 memposlength = length(obj.contour(index).cartesian); % length of the vector mempos
                % calculate the center position of the contour
% keyboard
                for i = 1:memposlength - 1
                    sumds(i) = ds(i) + ds(i + 1);
                end
                
                % to get the last part between the end and starting point of tracked
                % contour
                sumds(i + 1) = ds(i + 1) + ds(1);

                center(1) = (1/(2*L)) * mempos(:,1)' * sumds;
                center(2) = (1/(2*L)) * mempos(:,2)' * sumds;
                
                obj.contour(index).barioCenter = center;
                obj.contour(index).sumds = sumds;
% catch
%     keyboard
% end
            end
        end
        
        function obj = calculatePolarCoordinates(obj)
            % IMPORTANT: ADD OPTION TO C                                                                                                                                                                    ALCULATE USING MEAN CENTER OR
            % SIMILAR
            
            % transformation of coordinate origin to center of vesicle
%             nrOfContour = obj.getNrOfContours;
            nrOfContour = size(obj.cartesianX,2);
            polarCoordinates = cell(nrOfContour,1);

            goodContourIndexesSize = size(obj.goodContourIndexes);
            if goodContourIndexesSize(1)>1
                goodContourIndexesTMP = transpose(obj.goodContourIndexes);
            else
                goodContourIndexesTMP = obj.goodContourIndexes;
            end
            
%             for index = 1:nrOfContour
            for index = goodContourIndexesTMP
%             for index = 1:length(obj.contour)
%                 disp(['Transforming to polar coordinates ',num2str(index),' of ', num2str(nrOfContour)]);
%                 mempos(:,1) = mempos(:,1) * obj.parameterStruct.resolution;
                mempos = obj.getContourCartesian(index) * obj.parameterStruct.resolution;
%                 mempos = obj.contour(index).cartesian * obj.parameterStruct.resolution;
                center = obj.contour(index).referenceCenter * obj.parameterStruct.resolution;
                memposlength = length(mempos); % length of the vector mempos
                mempospol = zeros(memposlength,2); % vector containing the membrane position in polar coordinates

                mempos(:,1) = mempos(:,1) - center(1);
                mempos(:,2) = mempos(:,2) - center(2);

                disp(['Transforming to polar coordinates ',num2str(index),' of ', num2str(nrOfContour),' using center coordinate: ',num2str(center(1)),', ', num2str(center(2))]);
%                 disp(['using center coordinate: ',num2str(center(1)),', ', num2str(center(2))]);
    %             camino=sprintf('%s/contourscorregido/contour%i.txt',working_directory_path,j);
    %             save(camino,'mempos','-ascii'); % save the contour
% keyboard
                % TRANSFORMATION TO POLAR COORDINATES

                % with 
                % mempos(i,1) = r_i
                % mempos(i,2) = theta_i

                % matlab-version would be:
                % [mempos(:,2),mempos(:,1)]=cart2pol(mempos(:,1),mempos(:,2));

                % calculate radii (now the x_i and y_i are relative to center-position)
                for i = 1:memposlength
                    mempospol(i,1) = sqrt(mempos(i,1)^2 + mempos(i,2)^2);
                    
                    % calculate angles theta in range (-pi,pi]
                    if mempos(i,1) > 0
                    mempospol(i,2) = atan(mempos(i,2) / mempos(i,1));
                    elseif mempos(i,1) < 0 && mempos(i,2) >= 0
                        mempospol(i,2) = atan(mempos(i,2) / mempos(i,1)) + pi;
                    elseif mempos(i,1) < 0 && mempos(i,2) < 0
                        mempospol(i,2) = atan(mempos(i,2) / mempos(i,1)) - pi;
                    elseif mempos(i,1) == 0 && mempos(i,2) > 0
                        mempospol(i,1) = pi / 2;
                    elseif mempos(i,1) == 0 && mempos(i,2) < 0
                        mempospol(i,1) = - pi / 2;
                    end
                end % closes the angles calculation loop
                
                % change radius from pixels to meters
%                 mempospol(:,1) = mempospol(:,1) * obj.parameterStruct.resolution;

                polarCoordinates{index} = mempospol;
%                 obj.contour(index).polar = mempospol;
%                 % set mempos to the polar coordinates for further processing
%                 mempos = mempospol;
            end
        obj = obj.writePolarCoordinatesToArrayNew(polarCoordinates);
        end
            
        function obj = calculateFourierTransform(obj)
%             keyboard
            r0 = nan(size(obj.goodContourIndexes));
            a = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
            b = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
            c = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
            
%             for j = obj.goodContourIndexes
            goodContourIndexesSize = size(obj.goodContourIndexes);
            if goodContourIndexesSize(1) > goodContourIndexesSize(2)
                goodContourIndexesTmp = transpose(obj.goodContourIndexes);
            else
                goodContourIndexesTmp = obj.goodContourIndexes;
            end
            
            for j = goodContourIndexesTmp
%             for j = 1:obj.getNrOfContours
%             for j = 1:length(obj.contour)
                disp(['Fourier Transform Contour ',num2str(j),' of ', num2str(obj.getNrOfContours)]);
%                 disp(['Fourier Transform Contour ',num2str(j),' of ', num2str(length(obj.contour))]);
%                 mempos = obj.contour(j).cartesian;
                
%                 obj.contour(j).polar;
%                 mempos = obj.getContourPolar(j);
%                 [angles,radii] = obj.getContourPolarPhaseCorrected(j);
                [angles,radii] = obj.getContourPolarPhaseCorrectedInternal(j);
                
                mempos = nan(length(angles),2);
                mempos(:,1) = radii;
                mempos(:,2) = angles;
                
%                 mempos(:,1) = mempos(:,1) * obj.parameterStruct.resolution;
                memposlength = length(mempos);
%                 tmp = 0;
                tmp2 = 0;

                % calculate r0(j)
                for i = 1:memposlength - 1
                    if mempos(i + 1,2) * mempos(i,2) > 0
                    tmp = (mempos(i,1) + mempos(i + 1,1)) * abs(mempos(i + 1,2) - mempos(i,2));
                    else
                        tmp = (mempos(i,1) + mempos(i + 1,1)) * abs(mempos(i + 1,2) + mempos(i,2));
                    end
                    tmp2 = tmp2 + tmp;
                end
                % to get the last membrane part
                if mempos(i + 1,2) * mempos(i,2) > 0
                tmp2 = tmp2 + (mempos(i + 1,1) + mempos(1,1)) * abs(mempos(1,2) - mempos(i + 1,2));
                else
                    tmp2 = tmp2 + (mempos(i + 1,1) + mempos(1,1)) * abs(mempos(1,2) + mempos(i + 1,2));
                end

                r0(j) = (1/(4*pi)) * tmp2;
% keyboard
                % substract the mean radius of the vesicle to get the fluctuations
                % AS OF HERE mempos(:,1) will be the fluctuation of the membrane and NOT
                % the radius of the membrane position

                for i=1:memposlength
                   mempos(i,1) = mempos(i,1) - r0(j);
                end
%                 j;
                % save(sprintf('%s/memposs/mempos%i.txt',working_directory_path,j),'mempos','-ascii')
                
                for n = 1:obj.parameterStruct.nmax % loop to calculate the fourier coefficients 

                    % calculate a_n
%                     tmp = 0;
                    tmp2 = 0;

                    for i = 1:memposlength - 1
                        if mempos(i + 1,2) * mempos(i,2) > 0
                        tmp = (mempos(i,1) * cos(n*mempos(i,2)) + mempos(i + 1,1) * cos(n*mempos(i + 1,2))) * abs(mempos(i + 1,2) - mempos(i,2)) / 2;
                        else
                            tmp = (mempos(i,1) * cos(n*mempos(i,2)) + mempos(i + 1,1) * cos(n*mempos(i + 1,2))) * abs(mempos(i + 1,2) + mempos(i,2)) / 2;
                        end
                        tmp2 = tmp2 + tmp;
                    end
                    % to get the last membrane part
                    if mempos(1,2) * mempos(i + 1,2) > 0
                    tmp2 = tmp2 + (mempos(i + 1,1) * cos(n*mempos(i + 1,2)) + mempos(1,1) * cos(n*mempos(1,2))) * abs(mempos(1,2) - mempos(i + 1,2)) / 2;
                    else
                        tmp2 = tmp2 + (mempos(i + 1,1) * cos(n*mempos(i + 1,2)) + mempos(1,1) * cos(n*mempos(1,2))) * abs(mempos(1,2) + mempos(i + 1,2)) / 2;
                    end
                    
                    a(n,j) = (1/(pi*r0(j))) * tmp2;
                    
                    % calculate b_n
%                     tmp = 0;
                    tmp2 = 0;

                    for i = 1:memposlength - 1
                        if mempos(i + 1,2) * mempos(i,2) > 0
                        tmp = (mempos(i,1) * sin(n*mempos(i,2)) + mempos(i + 1,1) * sin(n*mempos(i + 1,2))) * abs(mempos(i + 1,2) - mempos(i,2)) / 2;
                        else
                            tmp = (mempos(i,1) * sin(n*mempos(i,2)) + mempos(i + 1,1) * sin(n*mempos(i + 1,2))) * abs(mempos(i + 1,2) + mempos(i,2)) / 2;
                        end
                        tmp2 = tmp2 + tmp;
                    end
                    % to get the last membrane part
                    if mempos(1,2) * mempos(i + 1,2) > 0
                        tmp2 = tmp2 + (mempos(i + 1,1) * sin(n*mempos(i + 1,2)) + mempos(1,1) * sin(n*mempos(1,2))) * abs(mempos(1,2) - mempos(i + 1,2)) / 2;
                    else
                        tmp2 = tmp2 + (mempos(i + 1,1) * sin(n*mempos(i + 1,2)) + mempos(1,1) * sin(n*mempos(1,2))) * abs(mempos(1,2) + mempos(i + 1,2)) / 2;  
                    end

                    b(n,j) = (1/(pi*r0(j))) * tmp2;

%                         c(n,j) = sqrt(a(n,j)^2 + b(n,j)^2);
                    c(n,j) = a(n,j) + 1i*b(n,j);
                end
                obj.contour(j).fourierTransform = c(:,j);
                obj.contour(j).radius = r0(j);
            end
            obj.fourierseries = c;
        end
        
        function r0 = calculateMode0radius(obj,mempos)
%                 tmp2 = 0;
%                 
%                 memposlength = length(mempos);
%                 
%                 % calculate r0(j)
%                 for i = 1:memposlength - 1
%                     if mempos(i + 1,2) * mempos(i,2) > 0
%                         tmp = (mempos(i,1) + mempos(i + 1,1)) * abs(mempos(i + 1,2) - mempos(i,2));
%                     else
%                         tmp = (mempos(i,1) + mempos(i + 1,1)) * abs(mempos(i + 1,2) + mempos(i,2));
%                     end
%                     tmp2 = tmp2 + tmp;
%                 end
% % keyboard
%                 % to get the last membrane part
%                 if mempos(i + 1,2) * mempos(i,2) > 0
%                     tmp2 = tmp2 + (mempos(i + 1,1) + mempos(1,1)) * abs(mempos(1,2) - mempos(i + 1,2));
%                 else
%                     tmp2 = tmp2 + (mempos(i + 1,1) + mempos(1,1)) * abs(mempos(1,2) + mempos(i + 1,2));
%                 end
%                 
%                 r0 = (1/(4*pi)) * tmp2;
                
                angles = mempos(:,2);
                radii = mempos(:,1);

                tmp3 = sum( ( radii(2:end) + radii(1:end-1) ) .* abs(angles(1:end-1) - angles(2:end)) );
%                 tmp3 = tmp3 + ( radii(end) + radii(1) ) .* abs(angles(end) - angles(1)); % we use '+' between the angles, because the angle changes signs here
                tmp3 = tmp3 + ( radii(end) + radii(1) ) .* abs(angles(end) + angles(1)); % we use '+' between the angles, because the angle changes signs here
                r0 = (1/(4*pi)) * tmp3;

%                 if abs(r0_new - r0)>1e-19
%                     keyboard
%                 end
        end
        
        function obj = calculateFourierTransformNEW(obj)
%             r0 = nan(size(obj.goodContourIndexes));
%             a = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
%             b = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
            c = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
            c_gpu = gpuArray(c);
            
            nrOfContours = obj.getNrOfContours;
            
            if size(obj.goodContourIndexes,1)>1
                obj.goodContourIndexes = transpose(obj.goodContourIndexes);
            end
            
            modeNrs = 1:obj.parameterStruct.nmax;
            modeNrs_gpu = gpuArray(modeNrs);
            
            for contourNr = obj.goodContourIndexes
                disp(['Fourier Transform Contour ',num2str(contourNr),' of ', num2str(nrOfContours)]);
                
                [angles,radii] = obj.getContourPolarPhaseCorrectedInternal(contourNr);

%                 [circumference,ds] = obj.calculateContourCircumference(contourNr);
%                 radius = (1/(4*pi)) * circumference * obj.parameterStruct.resolution;
% %                 r0(j) = (1/(4*pi)) * tmp2;

                mempos = nan(length(angles),2);
                mempos(:,1) = radii;
                mempos(:,2) = angles;
                
                radius = obj.calculateMode0radius(mempos);

% keyboard

%                 mempos(:,1) = mempos(:,1) * obj.parameterStruct.resolution;
%                 memposlength = length(mempos);
                
%                 mempos(i,1) = mempos(i,1) - r0(j);
                mempos(:,1) = mempos(:,1) - radius;
                radii = radii-radius;
                
                radius_gpu = gpuArray(radius);
                radii_gpu = gpuArray(radii);
                angles_gpu = gpuArray(angles);
% keyboard
%                 for modeNr = 1:obj.parameterStruct.nmax % loop to calculate the fourier coefficients 
%                     c(modeNr,contourNr) = a(modeNr,contourNr) + 1i*b(modeNr,contourNr);
%                     c(modeNr,contourNr) = obj.calculateFourierMode(modeNr,mempos,radius);
%                 end
                
% keyboard
%                 modeNr = transpose(1:obj.parameterStruct.nmax);
%                 c(:,contourNr) = arrayfun(@(modeNr) obj.calculateFourierMode(modeNr,mempos,radius),modeNr);
                
%                 c(:,contourNr) = obj.calculateFourierModeGpu(modeNrs,angles,radii,radius);

                c_gpu(:,contourNr) = obj.calculateFourierModeGpu(modeNrs_gpu,angles_gpu,radii_gpu,radius_gpu);
            end
            obj.fourierseries = gather(c_gpu);
        end
        
        function obj = calculateFourierTransformNEW2(obj)
        % this function is like 'calculateFourierTransformNEW', but without
        % using the GPU
        
%             r0 = nan(size(obj.goodContourIndexes));
%             a = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
%             b = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
            c = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
%             c_gpu = gpuArray(c);
            
            nrOfContours = obj.getNrOfContours;
            
            if size(obj.goodContourIndexes,1)>1
                obj.goodContourIndexes = transpose(obj.goodContourIndexes);
            end
            
            modeNrs = 1:obj.parameterStruct.nmax;
%             modeNrs_gpu = gpuArray(modeNrs);
            
            for contourNr = obj.goodContourIndexes
                disp(['Fourier Transform Contour ',num2str(contourNr),' of ', num2str(nrOfContours)]);
                
                [angles,radii] = obj.getContourPolarPhaseCorrectedInternal(contourNr);

%                 [circumference,ds] = obj.calculateContourCircumference(contourNr);
%                 radius = (1/(4*pi)) * circumference * obj.parameterStruct.resolution;
% %                 r0(j) = (1/(4*pi)) * tmp2;

                mempos = nan(length(angles),2);
                mempos(:,1) = radii;
                mempos(:,2) = angles;
                
                radius = obj.calculateMode0radius(mempos);

% keyboard

%                 mempos(:,1) = mempos(:,1) * obj.parameterStruct.resolution;
%                 memposlength = length(mempos);
                
%                 mempos(i,1) = mempos(i,1) - r0(j);
%                 mempos(:,1) = mempos(:,1) - radius;
                radii = radii-radius;
                
%                 radius_gpu = gpuArray(radius);
%                 radii_gpu = gpuArray(radii);
%                 angles_gpu = gpuArray(angles);
% keyboard
%                 for modeNr = 1:obj.parameterStruct.nmax % loop to calculate the fourier coefficients 
%                     c(modeNr,contourNr) = a(modeNr,contourNr) + 1i*b(modeNr,contourNr);
%                     c(modeNr,contourNr) = obj.calculateFourierMode(modeNr,mempos,radius);
%                 end
                
% keyboard
%                 modeNr = transpose(1:obj.parameterStruct.nmax);
%                 c(:,contourNr) = arrayfun(@(modeNr) obj.calculateFourierMode(modeNr,mempos,radius),modeNr);
                
%                 c(:,contourNr) = obj.calculateFourierModeGpu(modeNrs,angles,radii,radius);

                c(:,contourNr) = obj.calculateFourierModeNew(modeNrs,angles,radii,radius);
            end
            obj.fourierseries = c;
        end
        
        function obj = calculateFourierTransformNEW3(obj)
%             r0 = nan(size(obj.goodContourIndexes));
%             a = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
%             b = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
            c = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
            c_gpu = gpuArray(c);
            
            nrOfContours = obj.getNrOfContours;
            
            if size(obj.goodContourIndexes,1)>1
                obj.goodContourIndexes = transpose(obj.goodContourIndexes);
            end
            
            modeNrs = 1:obj.parameterStruct.nmax;
            modeNrs_gpu = gpuArray(modeNrs);
            
            counter = 1;
            for contourNr = obj.goodContourIndexes
%                 disp(['Fourier Transform Contour ',num2str(contourNr),' of ', num2str(nrOfContours)]);
                
                [angles,radii] = obj.getContourPolarPhaseCorrectedInternal(contourNr);

%                 [circumference,ds] = obj.calculateContourCircumference(contourNr);
%                 radius = (1/(4*pi)) * circumference * obj.parameterStruct.resolution;
% %                 r0(j) = (1/(4*pi)) * tmp2;

                mempos = nan(length(angles),2);
                mempos(:,1) = radii;
                mempos(:,2) = angles;
                
                radius(counter) = obj.calculateMode0radius(mempos);
                counter = counter + 1;
            end
            meanRadius = mean(radius);
% keyboard
            for contourNr = obj.goodContourIndexes
                disp(['Fourier Transform Contour ',num2str(contourNr),' of ', num2str(nrOfContours)]);
                
                [angles,radii] = obj.getContourPolarPhaseCorrectedInternal(contourNr);

%                 [circumference,ds] = obj.calculateContourCircumference(contourNr);
%                 radius = (1/(4*pi)) * circumference * obj.parameterStruct.resolution;
% %                 r0(j) = (1/(4*pi)) * tmp2;

%                 mempos = nan(length(angles),2);
%                 mempos(:,1) = radii;
%                 mempos(:,2) = angles;
                
%                 meanRadius = obj.calculateMode0radius(mempos);

% keyboard

%                 mempos(:,1) = mempos(:,1) * obj.parameterStruct.resolution;
%                 memposlength = length(mempos);
                
%                 mempos(i,1) = mempos(i,1) - r0(j);
%                 mempos(:,1) = mempos(:,1) - meanRadius;
                radii = radii-meanRadius;
% keyboard
                meanRadius_gpu = gpuArray(meanRadius);
                radii_gpu = gpuArray(radii);
                angles_gpu = gpuArray(angles);
% keyboard
%                 for modeNr = 1:obj.parameterStruct.nmax % loop to calculate the fourier coefficients 
%                     c(modeNr,contourNr) = a(modeNr,contourNr) + 1i*b(modeNr,contourNr);
%                     c(modeNr,contourNr) = obj.calculateFourierMode(modeNr,mempos,radius);
%                 end
                
% keyboard
%                 modeNr = transpose(1:obj.parameterStruct.nmax);
%                 c(:,contourNr) = arrayfun(@(modeNr) obj.calculateFourierMode(modeNr,mempos,radius),modeNr);
                
%                 c(:,contourNr) = obj.calculateFourierModeGpu(modeNrs,angles,radii,radius);

                c_gpu(:,contourNr) = obj.calculateFourierModeGpu(modeNrs_gpu,angles_gpu,radii_gpu,meanRadius_gpu);
            end
            obj.fourierseries = gather(c_gpu);
        end
        
        function obj = calculateFourierTransformFFT(obj)
%             r0 = nan(size(obj.goodContourIndexes));
%             a = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
%             b = nan(obj.parameterStruct.nmax,length(obj.goodContourIndexes));
% keyboard

            [angles,radii] = obj.getContourPolarPhaseCorrectedInternal(obj.goodContourIndexes(1));
            nrOfContourCoordinates = length(radii);
            c = nan(nrOfContourCoordinates,length(obj.goodContourIndexes));
            c_gpu = gpuArray(c);
            
            nrOfContours = obj.getNrOfContours;
            
            if size(obj.goodContourIndexes,1)>1
                obj.goodContourIndexes = transpose(obj.goodContourIndexes);
            end
            
            modeNrs = 1:nrOfContourCoordinates;
            modeNrs_gpu = gpuArray(modeNrs);
% keyboard
            % calculate mean radius
            %%
            meanRadius = 0;
            for contourNr = obj.goodContourIndexes
                disp(['Calculating mean contour radius using frame: ',num2str(contourNr),' of ', num2str(nrOfContours)]);
                
                [angles,radii] = obj.getContourPolarPhaseCorrectedInternal(contourNr);

%                 [circumference,ds] = obj.calculateContourCircumference(contourNr);
%                 radius = (1/(4*pi)) * circumference * obj.parameterStruct.resolution;
% %                 r0(j) = (1/(4*pi)) * tmp2;

                mempos = nan(length(angles),2);
                mempos(:,1) = radii;
                mempos(:,2) = angles;
                
                meanRadius = meanRadius + obj.calculateMode0radius(mempos);
            end
            meanRadius = meanRadius/length(obj.goodContourIndexes);
            
            %%
            radius = meanRadius;
            
            for contourNr = obj.goodContourIndexes
                disp(['Fourier Transform Contour ',num2str(contourNr),' of ', num2str(nrOfContours)]);
                
                [angles,radii] = obj.getContourPolarPhaseCorrectedInternal(contourNr);

%                 [circumference,ds] = obj.calculateContourCircumference(contourNr);
%                 radius = (1/(4*pi)) * circumference * obj.parameterStruct.resolution;
% %                 r0(j) = (1/(4*pi)) * tmp2;

                mempos = nan(length(angles),2);
                mempos(:,1) = radii;
                mempos(:,2) = angles;
                
%                 radius = obj.calculateMode0radius(mempos);

% keyboard

%                 mempos(:,1) = mempos(:,1) * obj.parameterStruct.resolution;
%                 memposlength = length(mempos);
                
%                 mempos(i,1) = mempos(i,1) - r0(j);
                mempos(:,1) = mempos(:,1) - radius;
                radii = radii-radius;
                
                radius_gpu = gpuArray(radius);
                radii_gpu = gpuArray(radii);
                angles_gpu = gpuArray(angles);
% keyboard
%                 for modeNr = 1:obj.parameterStruct.nmax % loop to calculate the fourier coefficients 
%                     c(modeNr,contourNr) = a(modeNr,contourNr) + 1i*b(modeNr,contourNr);
%                     c(modeNr,contourNr) = obj.calculateFourierMode(modeNr,mempos,radius);
%                 end
                
% keyboard
%                 modeNr = transpose(1:obj.parameterStruct.nmax);
%                 c(:,contourNr) = arrayfun(@(modeNr) obj.calculateFourierMode(modeNr,mempos,radius),modeNr);
                
%                 c(:,contourNr) = obj.calculateFourierModeGpu(modeNrs,angles,radii,radius);
% keyboard
%                 c_gpu(:,contourNr) = obj.calculateFourierModeGpu(modeNrs_gpu,angles_gpu,radii_gpu,radius_gpu);
                c_gpu(:,contourNr) = fft(radii_gpu);
                
%                 %%  
%                     x = 1:length(radii);
%                     interpFactor =  4;
%                     xInterp = linspace(1,length(radii),interpFactor*length(radii));
%                     radiiInterp = interp1(x,radii,xInterp);
%                     fftResult = fft(radii);
%                     fftInterpResult = fft(radiiInterp);
%                     win = hamming(size(radii,1));
%                     fftResultWithWindow = fft(radii.*win);
%                     %%
%                     plot(x,radii);
%                     hold on;
%                     plot(xInterp,radiiInterp,'r');
%                     hold off;
%                     %%
%                     hold on;
%                     plot(abs(fftResult));
%                     plot(abs(fftInterpResult),'c');
%                     plot(abs(fftResultWithWindow),'r');
%                     hold off;
%                     set(gca,'yscale','log');
%                 %%
            end
            obj.fourierseries = gather(c_gpu);
        end
        
        function c = calculateFourierModeGpu(obj,modeNrTmp,anglesTmp,radiiTmp,radius)
%             modeNr = 1;
            
%             modeNrTmp = modeNr;
%             radiiTmp = radii;
%             anglesTmp = angles;
            angles = repmat(anglesTmp,1,size(modeNrTmp,2));
            radii = repmat(radiiTmp,1,size(modeNrTmp,2));
            modeNr = repmat(modeNrTmp,size(anglesTmp),1);
            
            c = gpuArray(nan(1,size(modeNr,2)));
            
            while any(isnan(c)) % the while loop is necessary because the calculation on the GPU will sporatically cause some values of the array to be NaN, so that we then repeat the calculation; the cause for this is unknown, but it is likely a MATLAB bug, since it is not reproducible (one run it happens, the next one not); maybe it is caused by the GPU-threads not being synced correctly(?)
                % calculate a_n
                tmp3 = sum( (radii(1:end-1,:) .* cos(modeNr(2:end,:).*angles(1:end-1,:)) + radii(2:end,:) .* cos(modeNr(2:end,:).*angles(2:end,:))) .* abs(angles(2:end,:) - angles(1:end-1,:)) / 2, 1 );
%                 angleSum = sum(abs(angles(2:end,1) - angles(1:end-1,1)) / 2);
                tmp3 = tmp3 + (radii(end,:) .* cos(modeNr(1,:).*angles(end,:)) + radii(1,:) .* cos(modeNr(1,:).*angles(1,:))) .* abs(angles(1,:) + angles(end,:)) / 2;
%                 angleSum = angleSum + abs(angles(1,1) + angles(end,1)) / 2;
                a_new = (1/(pi*radius)) * tmp3;
    %             a_new = (1/(angleSum*radius)) * tmp3;

                % calculate b_n
                tmp3 = sum( (radii(1:end-1,:) .* sin(modeNr(2:end,:).*angles(1:end-1,:)) + radii(2:end,:) .* sin(modeNr(2:end,:).*angles(2:end,:))) .* abs(angles(2:end,:) - angles(1:end-1,:)) / 2, 1 );
                angleSum = sum(abs(angles(2:end,1) - angles(1:end-1,1)) / 2);
                tmp3 = tmp3 + (radii(end,:) .* sin(modeNr(1,:).*angles(end,:)) + radii(1,:) .* sin(modeNr(1,:).*angles(1,:))) .* abs(angles(1,:) + angles(end,:)) / 2;
                angleSum = angleSum + abs(angles(1,1) + angles(end,1)) / 2;
                b_new = (1/(pi*radius)) * tmp3;
    %             b_new = (1/(angleSum*radius)) * tmp3;
% keyboard
                c = a_new + 1i*b_new;
            end
            
            if  any(isnan(c))
                keyboard
            end
        end
        
        function c = calculateFourierModeNew(obj,modeNrTmp,anglesTmp,radiiTmp,radius)
            % this function is like calculateFourierModeGpu but without
            % using the GPU
%             modeNr = 1;
            
%             modeNrTmp = modeNr;
%             radiiTmp = radii;
%             anglesTmp = angles;
            angles = repmat(anglesTmp,1,size(modeNrTmp,2));
            radii = repmat(radiiTmp,1,size(modeNrTmp,2));
            modeNr = repmat(modeNrTmp,size(anglesTmp),1);

            % calculate a_n
            tmp3 = sum( (radii(1:end-1,:) .* cos(modeNr(2:end,:).*angles(1:end-1,:)) + radii(2:end,:) .* cos(modeNr(2:end,:).*angles(2:end,:))) .* abs(angles(2:end,:) - angles(1:end-1,:)) / 2, 1 );
%             angleSum = sum(abs(angles(2:end,1) - angles(1:end-1,1)) / 2);
            tmp3 = tmp3 + (radii(end,:) .* cos(modeNr(1,:).*angles(end,:)) + radii(1,:) .* cos(modeNr(1,:).*angles(1,:))) .* abs(angles(1,:) + angles(end,:)) / 2;
%             angleSum = angleSum + abs(angles(1,1) + angles(end,1)) / 2;
            a_new = (1/(pi*radius)) * tmp3;
%             a_new = (1/(angleSum*radius)) * tmp3;
                        
            % calculate b_n
            tmp3 = sum( (radii(1:end-1,:) .* sin(modeNr(2:end,:).*angles(1:end-1,:)) + radii(2:end,:) .* sin(modeNr(2:end,:).*angles(2:end,:))) .* abs(angles(2:end,:) - angles(1:end-1,:)) / 2, 1 );
%             angleSum = sum(abs(angles(2:end,1) - angles(1:end-1,1)) / 2);
            tmp3 = tmp3 + (radii(end,:) .* sin(modeNr(1,:).*angles(end,:)) + radii(1,:) .* sin(modeNr(1,:).*angles(1,:))) .* abs(angles(1,:) + angles(end,:)) / 2;
%             angleSum = angleSum + abs(angles(1,1) + angles(end,1)) / 2;
            b_new = (1/(pi*radius)) * tmp3;
%             b_new = (1/(angleSum*radius)) * tmp3;
% keyboard

            c = a_new + 1i*b_new;
            
            if any(any(isnan(c)))
                keyboard
            end
        end
        
        function c = calculateFourierMode(obj,modeNr,mempos,radius)
                
                % calculate a_n
% %                     tmp = 0;
%                 tmp2 = 0;
% 
% %                 for i = 1:memposlength - 1
%                 for i = 1:length(mempos) - 1
%                     if mempos(i + 1,2) * mempos(i,2) > 0
%                         tmp = (mempos(i,1) * cos(modeNr*mempos(i,2)) + mempos(i + 1,1) * cos(modeNr*mempos(i + 1,2))) * abs(mempos(i + 1,2) - mempos(i,2)) / 2;
%                     else
%                         tmp = (mempos(i,1) * cos(modeNr*mempos(i,2)) + mempos(i + 1,1) * cos(modeNr*mempos(i + 1,2))) * abs(mempos(i + 1,2) + mempos(i,2)) / 2;
%                     end
%                     tmp2 = tmp2 + tmp;
%                 end
% % keyboard
%                 % to get the last membrane part
%                 if mempos(1,2) * mempos(i + 1,2) > 0
%                     tmp2 = tmp2 + (mempos(i + 1,1) * cos(modeNr*mempos(i + 1,2)) + mempos(1,1) * cos(modeNr*mempos(1,2))) * abs(mempos(1,2) - mempos(i + 1,2)) / 2;
%                 else
%                     tmp2 = tmp2 + (mempos(i + 1,1) * cos(modeNr*mempos(i + 1,2)) + mempos(1,1) * cos(modeNr*mempos(1,2))) * abs(mempos(1,2) + mempos(i + 1,2)) / 2;
%                 end
% 
%                 a = (1/(pi*radius)) * tmp2;

                tmp3 = sum( (mempos(1:end-1,1) .* cos(modeNr*mempos(1:end-1,2)) + mempos(2:end,1) .* cos(modeNr*mempos(2:end,2))) .* abs(mempos(2:end,2) - mempos(1:end-1,2)) / 2 );
                tmp3 = tmp3 + (mempos(end,1) * cos(modeNr*mempos(end,2)) + mempos(1,1) * cos(modeNr*mempos(1,2))) * abs(mempos(1,2) + mempos(end,2)) / 2;
                a_new = (1/(pi*radius)) * tmp3;

               
                % calculate b_n
% %                     tmp = 0;
%                 tmp2 = 0;
%                 
% %                 for i = 1:memposlength - 1
%                 for i = 1:length(mempos) - 1
%                     if mempos(i + 1,2) * mempos(i,2) > 0
%                         tmp = (mempos(i,1) * sin(modeNr*mempos(i,2)) + mempos(i + 1,1) * sin(modeNr*mempos(i + 1,2))) * abs(mempos(i + 1,2) - mempos(i,2)) / 2;
%                     else
%                         tmp = (mempos(i,1) * sin(modeNr*mempos(i,2)) + mempos(i + 1,1) * sin(modeNr*mempos(i + 1,2))) * abs(mempos(i + 1,2) + mempos(i,2)) / 2;
%                     end
%                     tmp2 = tmp2 + tmp;
%                 end
% % keyboard
%                 % to get the last membrane part
%                 if mempos(1,2) * mempos(i + 1,2) > 0
%                     tmp2 = tmp2 + (mempos(i + 1,1) * sin(modeNr*mempos(i + 1,2)) + mempos(1,1) * sin(modeNr*mempos(1,2))) * abs(mempos(1,2) - mempos(i + 1,2)) / 2;
%                 else
%                     tmp2 = tmp2 + (mempos(i + 1,1) * sin(modeNr*mempos(i + 1,2)) + mempos(1,1) * sin(modeNr*mempos(1,2))) * abs(mempos(1,2) + mempos(i + 1,2)) / 2;  
%                 end
% 
%                 b = (1/(pi*radius)) * tmp2;

                tmp3 = sum( (mempos(1:end-1,1) .* sin(modeNr*mempos(1:end-1,2)) + mempos(2:end,1) .* sin(modeNr*mempos(2:end,2))) .* abs(mempos(2:end,2) - mempos(1:end-1,2)) / 2 );
%                 tmp3 = tmp3 + (mempos(i + 1,1) * sin(modeNr*mempos(i + 1,2)) + mempos(1,1) * sin(modeNr*mempos(1,2))) * abs(mempos(1,2) - mempos(i + 1,2)) / 2;
                tmp3 = tmp3 + (mempos(end,1) * sin(modeNr*mempos(end,2)) + mempos(1,1) * sin(modeNr*mempos(1,2))) * abs(mempos(1,2) + mempos(end,2)) / 2;

                b_new = (1/(pi*radius)) * tmp3;
%                         c(modeNr,j) = sqrt(a(modeNr,j)^2 + b(modeNr,j)^2);

%                 c = a + 1i*b;
                
                c = a_new + 1i*b_new;
        end
        
      
    end
end
