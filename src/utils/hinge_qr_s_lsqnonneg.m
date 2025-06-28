function [x,d,num_linsolve] = hinge_qr_s_lsqnonneg(A,b,tol)
% HINGE_S_LSQNONNEG   This function computes the nonnegative LSQ problem
%                       min_{x>=0} ||A*x-b||_{2}^{2}
%                       using the ``Method of Hinges" presented in
%                       ``A Simple New Algorithm for Quadratic Programming 
%                       with Applications in Statistics" by Mary C. Meyers.
%
%                   1) Suitable for sparse matrices.
%
%   Input
%           A           -   (m x n)-dimensional sparse matrix A
%           b           -   m-dimensional sparse col data vector.
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08).
%
%   Output
%           x   -   n-dimemsional solution vector to the NNLS problem.
%           d   -   m-dimensional col residual vector d = A*x-b
%           num_linsolve     - Number of times linsolve is called


%% Check if the matrix is sparse
if(~issparse(A))
    error("The matrix A is dense. " + ...
        "Use the function hinge_qr_lsqnonneg.m instead.")
end

%% Check if the LSQ is admissible;if so, return it
num_linsolve = 1;
u = A\b;
if(min(u) > -tol)
    x = u; 
    d = A*x-b;
    return
end


%% Compute the NNLS solutions via Meyers algorithm
[~, n] = size(A);
active_set = true(n,1);
while(true)
    % Check if the primal constraint is violated. If so, mark
    % the violating element. Else, compute the residual b - A*u.
    if(min(u) <= -tol)  
        x = zeros(n,1); 
        x(active_set) = u;
        [~,I] = min(x);
        active_set(I) = false;
    else    
        theta = A(:,active_set)*u;
        theta = b - theta;
        Atop_times_rho = (theta.'*A);
        [val,I] = max(Atop_times_rho);

        % Check if the dual constraint holds. If so, EXIT. Otherwise,
        % mark the corresponding columns to be added to the active set.
        if(val < tol)
            break;
        else
            active_set(I) = true;
        end
    end
    
    % Compute LSQ solution to A(:,active_set)*u = b.
    u = A(:,active_set)\b;
    num_linsolve = num_linsolve + 1;
end

x = zeros(n,1); x(active_set) = u;
d = -theta;
end