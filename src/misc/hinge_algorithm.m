function [x,p] = hinge_algorithm(A,y,active_set)
% HINGE_ALGORITHM   Compute the unique solution to the problem
%                   \min_{theta \in \Rn s.t. theta>=0} ||y-theta||^2
%                   subject to A*theta >=0
%                   using an active set method and warm start.
%
%               Tentative algorithm: Meyers' method of hinges
%
%   Input
%       A           -   (m x n) matrix
%       y           -   (n) dimensional col vector
%       active_set  -   m dimensional vector of true/false boolean values
%                       defining the initial active set
%
%   Output
%       x           -   Corresponding m-dimensional vector (polar cone?)
%       p           -   n dimensional col solution vector to the optimization
%                       problem.


%% Initialization
[m,~] = size(A);
tol = 1e-08;
flag = false;

% Polar cone edges + polar cone projection
b2 = -A*y;

% Check if there is at least one violated constraint
if(max(b2) > tol)
    [~,I] = max(b2);
    active_set(I) = true;
else
    flag = true;
end

% main loop
while(~flag)
    K = A(active_set,:);
    tmp = -(K*y);
    a = (K*K.')\tmp;

    if(min(a) < -tol)
        a_vec = zeros(m,1);
        a_vec(active_set) = a;
        [~,I] = min(a_vec);
        active_set(I) = false;
        flag = false;
    else
        theta = -(a.'*K).';
        rho = theta-y;
        b2 = A*rho;

        [val,I] = max(b2);
        if(val > tol)
            active_set(I) = true;
            flag = false;
        else
            flag = true;
        end
    end
end

a_vec = zeros(m,1);
a_vec(active_set) = a;
x = a_vec;
p = -rho;