function [x,d] = hinge_lsqnonneg(A,Q,R,u,b,tol)
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
%           A           -   (m x n)-dimensional design matrix A
%
%           Q           -   (m x n)-dimensional, economic form of the
%                           orthogonal matrix obtained from the QR
%                           decomposition [Q,R] = qr(A,"econ","vector")
%
%           R           -   (n x n)-dimensional, economic form of the
%                           upper triangular matrix obtained from the QR
%                           decomposition [Q,R] = qr(A,"econ","vector")
%
%           u           -   n-dimensional column vector, which is the
%                           least-squares solution to 
%                           min_{x \in \Rn} ||Ax - b||_{2}^{2}.
%           b           -   m-dimensional col data vector.
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08).
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
%   Note 2: Using QR for stability and QR column updates for speed.


%% Algorithm: Method of Hinges
% Invoke the method of hinges with full active set and the least-squares
% solution as a warm start.

[~, n] = size(A);
active_set = true(n,1);

while(true)
    if(min(u) < tol)
        x = zeros(n,1);
        x(active_set) = u;
        [~,I] = min(x);
        active_set(I) = false;
        remove_update = isscalar(I);
    else
        theta = Q*(R*u);
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
        [~, J] = min(u);

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
        % Note: In practice, the else statement is never reached?
    end

    % Compute the solution to Q*R = b.
    tmp = (b.'*Q).';
    u = R\tmp;
end

% Calculate final solution and its residual
x = zeros(n,1); x(active_set) = u;
d = -rho;
end