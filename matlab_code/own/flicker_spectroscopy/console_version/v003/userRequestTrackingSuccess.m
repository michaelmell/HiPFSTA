function trackingVariables = userRequestTrackingSuccess(trackingVariables)

% open question dialog of recognition
userResponse = questdlg('Was the contour correctly recognized?','Recognition Dialog','Yes','No','Cancel','No');
switch userResponse
    case {'No'}
        trackingVariables.firstdetection = 0;
    case {'Yes'}
        trackingVariables.firstdetection = 1;
    case {'Cancel'}
        delete(figure(1));
        trackingVariables.abort = true;
        return;
end