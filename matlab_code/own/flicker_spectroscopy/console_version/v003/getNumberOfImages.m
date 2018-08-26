function programParameters = getNumberOfImages(programParameters)

% check number of image files in image directory; the if-query is to select
% the adequate formatting of the file numeration
if programParameters.fileIndexDigits == 1
    programParameters.contourNr = 1;
    while exist([programParameters.image_data_directory_path,'/',programParameters.filename,num2str(programParameters.contourNr,'%01d'),'.tif']) == 2
       programParameters.contourNr = programParameters.contourNr + 1;
    end
    programParameters.contourNr = programParameters.contourNr-1;
    
    elseif programParameters.fileIndexDigits == 2
        programParameters.contourNr = 1;
    while exist([programParameters.image_data_directory_path,'/',programParameters.filename,num2str(programParameters.contourNr,'%02d'),'.tif']) == 2
       programParameters.contourNr = programParameters.contourNr + 1;
    end
    programParameters.contourNr = programParameters.contourNr-1;
        
    elseif programParameters.fileIndexDigits == 3
        programParameters.contourNr = 1;
    while exist([programParameters.image_data_directory_path,'/',programParameters.filename,num2str(programParameters.contourNr,'%03d'),'.tif']) == 2
       programParameters.contourNr = programParameters.contourNr + 1;
    end
    programParameters.contourNr = programParameters.contourNr-1;
    
    elseif programParameters.fileIndexDigits == 4
        programParameters.contourNr = 1;
    while exist([programParameters.image_data_directory_path,'/',programParameters.filename,num2str(programParameters.contourNr,'%04d'),'.tif']) == 2
       programParameters.contourNr = programParameters.contourNr + 1;
    end
    programParameters.contourNr = programParameters.contourNr-1;
end