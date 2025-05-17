function [u,d,eq_set,Q,R] = ...
    hinge_qr_homotopy(A,Q,R,b,eq_set,active_set,opts,tol)
% HINGE_HOMOTOPY    This function computes the nonnegative LSQ problem
%                   min_{x_active_set}>=0} ||A*x-b||_{2}^{2}
%                   using the ``Method of Hinges" presented in
%                   ``A Simple New Algorithm for Quadratic Programming 
%                   with Applications in Statistics" by Mary C. Meyers.
%
%                   This version of the Method of Hinges starts with the
%                   with the full active set.
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
%           active_set  -   subset of the equicorrelation set over which
%                       -   x(active_set) = 0.
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
%           Q   -   Updated orthogonal matrix from the QR decomposition
%           R   -   Updated upper triangular matrix from the QR
%                   decomposition


%% Algorithm: Method of Hinges (with initial full active set)
% Compute the least squares solution \w the QR decomposition
% Note: The dimension is already reduced to the equicorrelation set
[~, n] = size(A);
tmp = (b.'*Q(:,1:n)).';
u = linsolve(R(1:n,:),tmp,opts);

% Store the solution into its long form (n dimensions).
x = zeros(n,1);
x(active_set) = u;

% Check that the least-squares solution is positive over the active set.
if(min(x(active_set)) > -tol)
    tmp2 = R*u;
    d = Q*tmp2;
    d = d - b;
    return
end

% Compute the NNLS solutions via the Meyers' Method of Hinges
active_set = true(n,1);
while(true)
    remove_update = false;
    insert_update = false;
    x = zeros(n,1);
    x(active_set) = u;

    if(min(x(active_set)) <= -tol)      % Check if the primal constraint is violated.
        [~,I] = min(x);
        active_set(I) = false;
        remove_update = isscalar(I);
    else                    
        tmp = R*u;
        theta = Q*tmp;
        theta = b - theta;
        Atop_times_rho = (theta.'*A);
        [val,I] = max(Atop_times_rho);
        if(val >= tol)      % Check if the dual constraint is violated.
            active_set(I) = true;
            insert_update = isscalar(I);
        else
            break;          % Exit; All constraints are satisfied.
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
        % Extract new element added to the active set + column.
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

    % Check if the linsolve is singular; recompute QR decomposition if so.
    if(reciprocal < tol)
        disp("Warning: Linear system is nearly singular." + ...
            " Recomputing QR decomposition...")
        [Q,R] = qr(A(:,active_set));
        tmp = (rhs.'*Q).';
        u = linsolve(R,tmp,opts);
    end
end

% Update the equicorrelation set and compute its residual d = A*x-b
ind = find(eq_set);
eq_set(ind(~active_set)) = 0;
d = -theta;
end

% TODO: is it faster without calculating the reciprocal?