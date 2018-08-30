function returnStructure = loadToStructure(configFilePath)

returnStructure.init = 1; % initialize structure for tracking parameters

% load the parameters of from the configuration file
fid = fopen(configFilePath);
c = textscan(fid, '%s = %[^%\n]'); % loads config into the cell 'c' reading files upto 8Mbyte large
fclose(fid);

% write loaded cell do data-structure
for index = 1:length(c{1})-1
    eval(['returnStructure.', c{1}{index}, '=', c{2}{index},';']);
end