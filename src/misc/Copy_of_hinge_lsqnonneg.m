function [x,d] = hinge_lsqnonneg(A,b,tol)
% HINGE_LSQNONNEG   This function computes the nonnegative LSQ problem
%                   min_{x>=0} ||A*x-b||_{2}^{2}
%                   using the ``Method of Hinges" presented in
%                   ``A Simple New Algorithm for Quadratic Programming 
%                   with Applications in Statistics" by Mary C. Meyers
%
%   Input
%           A           -   (m x n)-dimensional design matrix A
%           b           -   m-dimensional col data vector
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08)
%
%   Output
%           x   -   n-dimemsional col solution vector to the NNLS problem
%           d   -   m-dimensional col residual vector d = A*x-b
%
%
%   Note 1: This version of Meyers' Method of Hinges starts with
%   active_set = {1,\dots,n\}, i.e., the active set starts full. This
%   becomes much faster than using MATLAB's lsqnonneg method.
%
%   Note 2: Using QR for stability and QR delete/updates.


%% Initialization
% Solve the system Ax = b using the QR decomposition.
[Q,R] = qr(A,"econ","vector");
tmp = (b.'*Q).';
a = R\tmp;
rho = b - Q*(R*a);

% Check if the solution is positive. If true, we are done. Else, proceed.
if(min(a) > tol)
    x = a;
    d = -rho;
    return
end


%% Algorithm: Method of Hinges
% If the least-square solution has a negative component, proceed
% according the Method of Hinges.

[~, n] = size(A);
active_set = true(n,1);

while(true)
    if(min(a) < -tol)
        x = zeros(n,1);
        x(active_set) = a;
        [~,I] = min(x);
        active_set(I) = false;
        remove_update = isscalar(I);
    else
        theta = Q*(R*a);
        rho = b - theta;
        [val,I] = max((rho.'*A));

        if(val > tol)
            active_set(I) = true;
            remove_update = false;
        else
            break;
        end
    end

    % Update the QR decomposition. If removing one column, use qrdelete.
    % Otherwise, recompute the QR decomposition.
    if(remove_update)
        [~, J] = min(a);

        % The following is equivalent to [Q,R] = qrdelete(Q,R,J);
        % Some overhead has been removed to optimize for speed.
        [~,nq] = size(Q);
        [meff,~] = size(R);
        R(:,J) = [];
        [Q,R] = matlab.internal.math.deleteCol(Q,R,J);
        R(meff,:)=[];
        Q(:,nq)=[];
    else
        [Q,R] = qr(A(:,active_set),"econ","vector");
    end

    % Compute the solution to Q*R = b.
    tmp = (b.'*Q).';
    a = R\tmp;
end

% Calculate final solution and its residual
x = zeros(n,1); x(active_set) = a;
d = -rho;
end