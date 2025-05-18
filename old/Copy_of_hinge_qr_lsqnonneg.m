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


%% Algorithm: Method of Hinges (with initial full active set)
% Compute the least squares solution. If positive, then return.
[~, n] = size(A);
active_set = true(n,1);
tmp = (b.'*Q).';
[u,r1] = linsolve(R,tmp,opts);

% Check if the system is singular. If it is, avoid using QR.
if(r1 < tol)
    disp('debug')
    [x,d] = hinge_lsqnonneg(A,b,tol);
    active_set(x==0) = false;
    [Q,R] = qr(A(:,active_set));

    ind = find(eq_set);
    eq_set(ind(~active_set)) = 0;
    u = x(x~=0);
    return;
end

if(min(u) > -tol)
    tmp2 = R*u;
    d = Q*tmp2;
    d = d - b;
    return
end

% Compute the NNLS solutions via the Meyers' Method of Hinges.
while(true)
    remove_update = false;
    insert_update = false;

    % Check if the primal constraint is violated. If not, compute the
    % residual b - A*u.
    if(min(u) <= -tol)  
        x = zeros(n,1); 
        x(active_set) = u;
        [~,I] = min(x);
        active_set(I) = false;
        remove_update = isscalar(I);
    else    
        tmp = R*u;
        theta = Q*tmp;
        theta = b - theta;
        Atop_times_rho = (theta.'*A);
        [val,I] = max(Atop_times_rho);

        % Check if the dual constraint holds. If so, exit. Otherwise,
        % mark the corresponding columns to be added to the active set.
        if(val < tol)
            break;
        else
            active_set(I) = true;
            insert_update = isscalar(I);
        end
    end

    % Update the QR decomposition.
    if(remove_update)
        % Execute [Q,R] = qrdelete(Q,R,J) without overhead.
        [~, J] = min(u);
        R(:,J) = [];
        [Q,R] = matlab.internal.math.deleteCol(Q,R,J);
    elseif(insert_update)
        % Extract new element added to the active set and its column.
        [~, J] = max(Atop_times_rho);
        col = A(:,J);
        loc = find(find(active_set) == J);
        
        % Execute [Q,R] = qrinsert(Q,R,loc,col) without overhead.
        [~,nr] = size(R);
        R(:,loc+1:nr+1) = R(:,loc:nr);
        R(:,loc) = (col.'*Q).';
        [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
    else
        % Multiple column deletes or inserts not supported.
        [Q,R] = qr(A(:,active_set));
    end

    % Compute LSQ solution to Au = b with the QR decomposition A=Q*R.
    tmp = (b.'*Q).';
    [u,r2] = linsolve(R,tmp,opts);

    % Check if the linsolve is singular; recompute QR decomposition if so.
    if(r2 < tol)
        disp("Warning: Linear system is nearly singular." + ...
            " Recomputing QR decomposition...")
        [Q,R] = qr(A(:,active_set));
        tmp = (b.'*Q).';
        u = linsolve(R,tmp,opts);
    end    
end

% Update the equicorrelation set and compute its residual d = A*x-b.
ind = find(eq_set);
eq_set(ind(~active_set)) = 0;
d = -theta;
end