function returnStructure = setProgramParameters(parameterStruct)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% setProgramParameters    Function for writing the parameters from the
%%% structure 'parameterStruct' to the structure 'returnStructure', which
%%% will be return to the structure 'programParameters'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% returnStructure.program_directory_path = parameterStruct.program_directory_path;
returnStructure.filename = parameterStruct.filename;
returnStructure.data_analysis_directory_path = parameterStruct.data_analysis_directory_path;
% returnStructure.data_analysis_configFileName = parameterStruct.data_analysis_configFileName;
returnStructure.image_data_directory_path = parameterStruct.image_data_directory_path;
returnStructure.fileIndexDigits = parameterStruct.fileIndexDigits;
returnStructure.membrane_thickness = parameterStruct.membrane_thickness;