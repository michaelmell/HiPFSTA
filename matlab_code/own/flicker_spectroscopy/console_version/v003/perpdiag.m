function B=perpdiag(A)
% function the gives back the diagonal perpendicular to the main diagonal;
% it only works for square matrixes (at the moment)

s=size(A);

B=A(s(1):s(1)-1:s(1)*s(2)-1);





% this was the example from which I learned; gets the main diagonal
% s=size(A);
% 
% if s(1)<=s(2)
%     B=A(1:s(1)+1:s(1)^2);
%     return;
% else
%     B=A(1:s(1)+1:s(1)*s(2));
% end