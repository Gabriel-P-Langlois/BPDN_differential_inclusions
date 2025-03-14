function [u,d,eq_set,Q,R] = ...
    hinge_qr_lsqnonneg(A,Q,R,b,eq_set,opts,tol)
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
%           Q           -   (m x n)-dimensional, economic form of the
%                           orthogonal matrix obtained from the QR
%                           decomposition [Q,R] = qr(A,"econ","vector")
%           R           -   (n x n)-dimensional, economic form of the
%                           upper triangular matrix obtained from the QR
%                           decomposition [Q,R] = qr(A,"econ","vector")
%                           min_{x \in \Rn} ||Ax - b||_{2}^{2}.
%           b           -   m-dimensional col data vector.
%           eq_set      -   equicorrelation set of the BPDN problem
%           opts        -   opts field for the linsolve function.
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08).
%
%   Output
%           u   -   neff-dimemsional col solution vector of nonzero
%                   coefficients to the NNLS problem, where neff = 
%                   sum(non-zero components to the NNLS solution).
%           d   -   m-dimensional col residual vector d = A(:,neff)*x-b
%           eq_set      -   Updated equicorrelation set of the BPDN problem
%                           It is is strictly ``less" than the input.
%           Q   -   Updated orthogonal matrix from the QR decomposition
%           R   -   Updated upper triangular matrix from the QR
%                   decomposition
%
%
%   Note: This version of Meyers' Method of Hinges starts with
%   active_set = {1,\dots,n\}, i.e., the active set is full.


%% Algorithm: Method of Hinges \w initial full active set
% Compute the least squares solution \w the QR decomposition
% and check that it's nonnegative. 
[~, n] = size(A);
tmp = (b.'*Q(:,1:n)).';
u = linsolve(R(1:n,:),tmp,opts);
if(min(u) > -tol)
    tmp2 = R*u;
    d = Q*tmp2;
    d = d - b;
    return
end

% Invoke the Meyers' Method of Hinges \w full active set to compute
% the NNLS solution.
active_set = true(n,1);
while(true)
    remove_update = false;
    insert_update = false;

    if(min(u) <= -tol)      % Check if the primal constraint is satisfied
        x = zeros(n,1);
        x(active_set) = u;
        [~,I] = min(x);
        active_set(I) = false;
        remove_update = isscalar(I);
    
    else                    % Check if the dual constraint is satisfied
        tmp = R*u;
        theta = Q*tmp;
        theta = b - theta;
        Atop_times_rho = (theta.'*A);
        [val,I] = max(Atop_times_rho);
        if(val >= tol)   
            active_set(I) = true;
            insert_update = isscalar(I);
        else
            break;      % Exit; All constraints are satisfied.
        end
    end


    % Update the QR decomposition. If removing one column, use qrdelete.
    % If adding one column, use qrinsert. Otherwise, reupdate.
    if(remove_update)
        % Execute [Q,R] = qrdelete(Q,R,J) without overhead.
        [~, J] = min(u);
        R(:,J) = [];
        [Q,R] = matlab.internal.math.deleteCol(Q,R,J);
        
    elseif(insert_update)
        % Extract new element added to the active set + column
        [~, J] = max(Atop_times_rho);
        col = A(:,J);
        
        % Execute [Q,R] = qrinser(Q,R,J,col) without overhead.
        [~,nr] = size(R);
        R(:,J+1:nr+1) = R(:,J:nr);
        R(:,J) = (col.'*Q).';
        [Q,R] = matlab.internal.math.insertCol(Q,R,J);
    else
        % Multiple columns delete or inserts not supported.
        [Q,R] = qr(A(:,active_set));
    end

    % Compute LSQ solution to QRu = b.
    [~,neff] = size(R);
    tmp = (b.'*Q(:,1:neff)).';
    [u,reciprocal] = linsolve(R(1:neff,:),tmp,opts);

    % Check if the linear solve is singular. If so,
    % recompute the QR decomposition from scratch.
    if(reciprocal < tol)
        disp("Warning: Linear system is nearly singular. Recomputing QR decomposition")
        [Q,R] = qr(A(:,active_set));
        tmp = (rhs.'*Q).';
        u = linsolve(R,tmp,opts);
    end
end

% Update the equicorrelation set.
ind = find(eq_set);
eq_set(ind(~active_set)) = 0;

% Compute the residual d = A*x-b \equiv -theta
d = -theta;
end