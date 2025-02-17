function [x,d] = hinge_lsqnonneg(A,b,active_set,tol)
% HINGE_LSQNONNEG   This function computes the nonnegative LSQ problem
%                   min_{x>=0} ||A*x-b||_{2}^{2}
%                   using the ``Method of Hinges" presented in
%                   ``A Simple New Algorithm for Quadratic Programming 
%                   with Applications in Statistics" by Mary C. Meyers
%
%   Input
%           A           -   (m x n)-dimensional design matrix A
%           b           -   m-dimensional col data vector
%           active_set  -   n-dim col vector of boolean values;
%                           this is the initial set of
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08)
%
%   Output
%           x   -   n-dimemsional col solution vector to the NNLS problem
%           d   -   m-dimensional col residual vector d = A*x-b
%
%


%% Initialization
[~,n] = size(A);
flag = false;

% Polar cone edges + polar cone projection
b2 = (b.'*A).';

% Check if there is at least one violated constraint
[val, I] = max(b2);
if(val > tol)
    active_set(I) = true;
else
    flag = true;
end

% repeat until convergence
while(~flag)
    K = A(:,active_set);
    M = K.'*K;
    tmp = (b.'*K).';
    a = M\tmp;

    if(min(a) < -tol)
        x = zeros(n,1);
        x(active_set) = a;
        [~,I] = min(x);
        active_set(I) = false;
        flag = false;
    else
        theta = K*a;
        rho = b - theta;
        b2 = (rho.'*A).';

        [val,I] = max(b2);
        if(val > tol)
            active_set(I) = true;
            flag = false;
        else
            flag = true;
        end
    end
end


%% Calculate final solution and its residual
x = zeros(n,1); x(active_set) = a;
d = -rho;

end