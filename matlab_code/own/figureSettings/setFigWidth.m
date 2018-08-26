function setFigWidth(varargin)

Args = struct('width', 'singlecolumn', ...
              'height', [], ...
              'handle', [] ...
            );

Args = parseArgs(varargin,Args);

setVariables( Args )

if isempty( hfig )
    hfig = gcf;
end

switch width
    case {'singlecolumn','fullpagewidth'}
        posWidth = 700; % in px
        posHeight = 500; % in px
    case {'twocolumn','halfpagewidth'}
        posWidth = 450; % in px
        posHeight = 350; % in px
end

if isempty( height )
    height = 'fullheight';
end

switch height
    case 'fullheight'
        % do nothing; value as set above
    case 'halfheight'
        posHeight = posHeight/2; % in px
    case 'thirdheight'
        posHeight = posHeight/3; % in px
    case 'twothirdsheight'
        posHeight = posHeight*2/3; % in px
    case 'doubleheight'
        posHeight = posHeight*2; % in px
    case 'tripleheight'
        posHeight = posHeight*3; % in px
end


fpos = get(hfig,'Position');
set(hfig, 'Position', [fpos(1) fpos(2) posWidth posHeight]);

% switch width
%     case 'singlecolumn'
%         set(hfig, [fpos(1) fpos(2) posWidth posHeight])
%     case 'twocolumn'
%     case 'threecolumn'
% end


%% Functions
function setVariables( Args )
    for tmp = transpose( fieldnames( Args ) )
        name = tmp{1};
        if strcmp(name,'handle')
            assignin('caller','hfig',Args.('handle'));
        else
            assignin('caller',name,Args.(name));
        end
    end
