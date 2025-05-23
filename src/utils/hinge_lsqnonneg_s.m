function [x,d] = hinge_lsqnonneg_s(A,b,tol)
% HINGE_LSQNONNEG   This function computes the nonnegative LSQ problem
%                   min_{x>=0} ||A*x-b||_{2}^{2}
%                   using the ``Method of Hinges" presented in
%                   ``A Simple New Algorithm for Quadratic Programming 
%                   with Applications in Statistics" by Mary C. Meyers.
%
%                   This version of the Method of Hinges starts with the
%                   with the full active set. This requires first computing
%                   the least-squares solution u = (A.'*A)^{-1}(A.'*b),
%                   which is given as input to this algorithm.
%
%   Input
%           A           -   (m x n)-dimensional sparse design matrix A
%           b           -   m-dimensional col data vector.
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08).
%
%   Output
%           x   -   n-dimemsional solution vector to the NNLS problem.
%           d   -   m-dimensional residual vector d = A*x-b


%% Algorithm: Method of Hinges
% Invoke the method of hinges with full active set
[~, n] = size(A);
eqset = true(n,1);
u = A\b;

% Check if the least-squares solution is feasible. If so, we are done.
if(min(u) >= -tol)
    d = A*u - b;
    x = u;
    return;
end

% Compute the solution via the method of hinges
while(true)
    if(min(u) < -tol)       % Check if the primal constraint is violated.
        x = zeros(n,1);
        x(eqset) = u;
        [~,I] = min(x);
        eqset(I) = false;
    else                    % Check if the dual constraint is violated.
        theta = A(:,eqset)*u;
        rho = b - theta;
        [val,I] = max((rho.'*A));

        if(val > tol)
            eqset(I) = true;
        else
            break;          % STOP; all constraints are satisfied.
        end
    end

    % Compute the solution to A(:,eqset)*u = b and continue.
    u = A(:,eqset)\b;
end

% Return the solution vector x and the residual d = A*x-b
x = zeros(n,1);
x(eqset) = u;
d = -rho;
end