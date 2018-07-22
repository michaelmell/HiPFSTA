function p = linearFit(x,y)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% linearFit   Function to do linear fit
%%% The function copies the MATLAB-own function POLYFIT, which is for
%%% general fitting of polynomials, but was reduced to only fit linear
%%% polynomials.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % transform coordinates to column vectors
% x = x(:);
% y = y(:);
% 
% % TODO: optimize this!
% x_y_length = length(x);
% 
% % Construct Vandermonde matrix.
% V(:,2) = ones(x_y_length,1);
% 
% V(:,1) = x.*V(:,2);
% 
% p = V\y;
% 
% keyboard

%%% version from forum
x_mean = meanOneDimension(x);
y_mean = meanOneDimension(y);

% find b1 and b2 in the equation:
% y = b1*x + b0
b1 = sum((x-x_mean).*(y-y_mean)) / sum((x-x_mean).^2);
b0 = y_mean - b1*x_mean;

p(2) = b0;
p(1) = b1;