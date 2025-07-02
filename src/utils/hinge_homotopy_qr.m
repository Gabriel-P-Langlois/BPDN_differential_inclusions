function [x,d,num_linsolve] = ...
    hinge_homotopy_qr(A,Q,R,b,ind,opts,tol)
% HINGE_homotopy_qr   This function computes the nonnegative LSQ problem
%                       min_{x \in \Rn} ||A*x-b||_{2}^{2}
%                       subject to xj >= 0 if j \in ind 
%                       using the ``Method of Hinges" presented in
%                       ``A Simple New Algorithm for Quadratic Programming 
%                       with Applications in Statistics" by Mary C. Meyers.
%
%
%
%   Input
%           A           -   (m x l)-dimensional design matrix A
%           Q           -   (m x n)-dimensional, non-economic
%                           orthogonal matrix obtained from the QR
%                           decomposition [Q,R] = qr(A).
%           R           -   (n x n)-dimensional, economic form of the
%                           upper triangular matrix obtained from the QR
%                           decomposition [Q,R] = qr(A,"econ","vector")
%           b           -   m-dimensional col data vector.
%           ind         -   ind \coloneqq \subset {1,...l}
%           opts        -   opts field for the linsolve function.
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08).
%
%   Output
%           x   -   l-dimensional col solution vector of nonzero
%                   coefficients to the NNLS problem.
%           d   -   m-dimensional col residual vector d = A*x - b
%           num_linsolve     - Number of times linsolve is called


%% Initialization
if(issparse(A))
    error("The matrix A is sparse. " + ...
        "Use [...] instead.")
end
[~, n] = size(A);
active_set = true(n,1);


%% Check if the LSQ is admissible; if so, return it
tmp = Q.'*b;
u = linsolve(R,tmp,opts);
num_linsolve = 1;
x = u;
d = A*u - b;

if(isempty(ind) || min(u(ind)) > tol)
    return;
end


%% Compute the NNLS solutions via Meyers algorithm
while(true)

    if(min(x(ind)) < -tol)  % Check if the primal constraint is violated.
        [~,J] = min(x(ind));
        I = ind(J);
        active_set(I) = false;
    else                    % Check if the dual constraint is violated.
        theta = A(:,active_set)*u;
        rho = b - theta;
        tmp2 = rho.'*A;
        [val,J] = max(tmp2(ind));
        I = ind(J);
        if(val > tol)
            active_set(I) = true;
        else
            break;
        end
    end

    % Compute the solution to A(:,eqset)*u = b and continue.
        [Q,R] = qr(A(:,active_set),'econ');
    tmp = Q.'*b;
    [u,~] = linsolve(R,tmp,opts);
    num_linsolve = num_linsolve + 1;

    x = zeros(n,1); x(active_set) = u;
    d = A(:,active_set)*u-b;
end
end