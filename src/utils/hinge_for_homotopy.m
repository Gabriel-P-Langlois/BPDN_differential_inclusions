function [x,d] = hinge_for_homotopy(A,b,ind,tol)
% HINGE_MOD_LSQNONNEG   This function computes the nonnegative LSQ problem
%                       min_{x \in \Rn} ||A*x-b||_{2}^{2}
%                       subject to xj >= 0 if j \in ind 
%                       using the ``Method of Hinges" presented in
%                       ``A Simple New Algorithm for Quadratic Programming 
%                       with Applications in Statistics" by Mary C. Meyers.
%
%                       This algorithm is tailored to the homotopy method
%                       for the basis pursuit denoising problem.
%
%
%   Input
%           A           -   (m x l)-dimensional design matrix A
%           b           -   m-dimensional col data vector.
%           ind         -   ind \coloneqq \subset {1,...l}
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08).
%
%   Output
%           x   -   l-dimensional col solution vector of nonzero
%                   coefficients to the NNLS problem.
%           d   -   m-dimensional col residual vector d = A*x - b


%% Algorithm: Method of Hinges
% Initialization + compute LSQ problem with full active set
[~, l] = size(A);
eqset = true(l,1);

x = zeros(l,1);
[u,~] = linsolve(A,b);
x(eqset) = u;
d = A*u - b;

% Check if the LSQ solution is feasible. If so, stop -- it is optimal.
if(isempty(ind) || min(u(ind)) > tol)
    return;
end

% Compute the solution via the method of hinges
while(true)
    if(min(x(ind)) < -tol)  % Check if the primal constraint is violated.
        [~,J] = min(x(ind));
        I = ind(J);
        eqset(I) = false;
    else                    % Check if the dual constraint is violated.
        theta = A(:,eqset)*u;
        rho = b - theta;
        tmp = rho.'*A;
        [val,J] = max(tmp(ind));
        I = ind(J);
        if(val > tol)
            eqset(I) = true;
        else
            break;
        end
    end

    % Compute the solution to A(:,eqset)*u = b and continue.
    x = zeros(l,1);
    [u,~] = linsolve(A(:,eqset),b);
    x(eqset) = u;
    d = A(:,eqset)*u-b;
end
end