function prox = l1_prox_operator(x,t)
% l1_prox_operator  Compute the soft thresholding operator
%                   of x with parameter t
%
%   Input
%       x       -   n dimensional row or column vector
%       t       -   positive scalar
%
%   Output
%       prox    -   n dimensional row or column vector

% Soft thresholding operator
prox = abs(x)-t;
prox = sign(x).*(prox+abs(prox))*0.5;

end