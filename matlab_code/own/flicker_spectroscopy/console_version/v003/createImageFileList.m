function parameterStruct = createImageFileList(parameterStruct)

% obj.parameterStruct.datasetLoadPath = path;
% parameterStruct.image_data_directory_path
fileList = dir([parameterStruct.image_data_directory_path,'/*',parameterStruct.filePostfix]);
parameterStruct.image_file_list = {fileList.name};
