function [contourclosed,breakofpixel] = checkBreakOffCondition(index2,trackingParameters,trackingVariables)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Break condition of the algorithm:
%%% This compares the last calculated pixelposition with the pixel position 
%%% (x_10,y_10) at index2 = trackingParameters.comparepixel up to
%%% index2 = trackingParameters.comparepixel+trackingParameters.nrofcomparepixel.
%%% If one of the conincides the vesicle tracing is complete.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

breakofpixel = 1; % the variable  "breakofpixel" will be used below to eliminate 
                  % the first certain number membrane positions that were traced twice
contourclosed = 0;

if index2>trackingParameters.comparepixel+trackingParameters.nrofcomparepixel
    for breakofpixel = trackingParameters.comparepixel:trackingParameters.comparepixel+trackingParameters.nrofcomparepixel 
        if trackingVariables.pixpos(breakofpixel,:)==trackingVariables.pixpos(index2+1,:)
            a = trackingVariables.xymempos(10,:)-trackingVariables.xymempos(1,:);
            b = trackingVariables.xymempos(index2,:)-trackingVariables.xymempos(index2-9,:);
            % calculate angle:
            if norm(a) ~= 0 && norm(b) ~= 0
                alpha = acos((a*transpose(b))/(norm(a)*norm(b)));
                maxangle = 0.7;
                if alpha<maxangle
                    contourclosed = 1; % set break-off Variable to 1
                    break;
                end
            end
        end
    end
end

trackingVariables.breakofpixel = breakofpixel;