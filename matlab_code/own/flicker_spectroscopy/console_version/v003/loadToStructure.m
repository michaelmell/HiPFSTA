function returnStructure = loadToStructure(configFilePath)

returnStructure.init = 1; % initialize structure for tracking parameters

% load the parameters of from the configuration file
fid = fopen(configFilePath);
c = textscan(fid, '%s = %[^%\n]'); % loads config into the cell 'c' reading files upto 8Mbyte large
fclose(fid);

% write loaded cell do data-structure
if(isOctave())
    maxIndex = length(c{1})-1; % for some reason Octave requires the index to only go to length-1
else
    maxIndex = length(c{1});
end

for index = 1:maxIndex
    eval(['returnStructure.', c{1}{index}, '=', c{2}{index},';']);
end

%%
%% Return: true if the environment is Octave.
%%
function retval = isOctave
  persistent cacheval;  % speeds up repeated calls

  if isempty (cacheval)
    cacheval = (exist ('OCTAVE_VERSION', 'builtin') > 0);
  end

  retval = cacheval;
end
end