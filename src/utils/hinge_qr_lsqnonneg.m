function [x,d,eq_set,Q,R,num_linsolve] = ...
    hinge_qr_lsqnonneg(A,Q,R,b,eq_set,opts,tol)
% HINGE_QR_LSQNONNEG   This function computes the nonnegative LSQ problem
%                       min_{x>=0} ||A*x-b||_{2}^{2}
%                       using the ``Method of Hinges" presented in
%                       ``A Simple New Algorithm for Quadratic Programming 
%                       with Applications in Statistics" by Mary C. Meyers.
%
%                   This version of the Method of Hinges starts with the
%                   with the full active set and maintains the QR
%                   decomposition of A as we remove elements from the
%                   active set or add elements to the active set.
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
%                           NOTE: Used to improve performance, not actually
%                           recompute the equicorrelation set.
%           opts        -   opts field for the linsolve function.
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08).
%
%   Output
%           x   -   n-dimemsional solution vector to the NNLS problem.
%           d   -   m-dimensional col residual vector d = A*x-b
%           eq_set      -   Updated equicorrelation set of the BPDN problem
%           Q   -   Updated orthogonal matrix from the QR decomposition
%           R   -   Updated upper triangular matrix from the QR
%                   decomposition
%           num_linsolve     - Number of times linsolve is called


%% Initialization
[~, n] = size(A);
num_linsolve = 0;
active_set = true(n,1);
tmp = Q.'*b;


%% Check if the LSQ is admissible and identify potential issues
if(~issparse(A))
    [u,r1] = linsolve(R,tmp,opts);
    num_linsolve = num_linsolve + 1;
    
    % Check if the system is singular. If so, avoid solving it with QR.
    if(r1 < tol)
        [x,d] = hinge_lsqnonneg(A,b,tol);
        active_set(x==0) = false;
        [Q,R] = qr(A(:,active_set));
        ind = find(eq_set);
        eq_set(ind(~active_set)) = 0;
        return;
    end
else
    u = R\tmp;
    num_linsolve = num_linsolve + 1;
end

% Check if the LSQ solution is admissible.
if(min(u) > -tol)
    x = u; 
    d = A*x-b;
    return
end


%% Compute the NNLS solutions via Meyers algorithm
while(true)
    remove_update = false;
    insert_update = false;

    % Check if the primal constraint is violated. If so, mark
    % the violating element. Else, compute the residual b - A*u.
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

        % Check if the dual constraint holds. If so, EXIT. Otherwise,
        % mark the corresponding columns to be added to the active set.
        if(val < tol)
            break;
        else
            active_set(I) = true;
            insert_update = isscalar(I);
        end
    end

    % Update the QR decomposition.
    if(~issparse(A))
        if(remove_update)
            [~, J] = min(u);
            [Q,R] = qrdelete(Q,R,J);
        elseif(insert_update)
            % Extract the new element added to the active set and its column.
            [~, J] = max(Atop_times_rho);
            col = A(:,J);
            loc = find(find(active_set) == J);
            [Q,R] = qrinsert(Q,R,loc,col);
        else
            % Multiple column deletes or inserts not supported.
            [Q,R] = qr(A(:,active_set));
        end
    else
        [Q,R] = qr(A(:,active_set));
    end
    
    % Compute LSQ solution to Au = b with the QR decomposition A=Q*R.
    tmp = Q.'*b;
    if(~issparse(A))
        [u,r2] = linsolve(R,tmp,opts);
        num_linsolve = num_linsolve + 1;
        % Check if linsolve is singular; recompute QR decomposition if so.
        if(r2 < tol)
            disp("Warning: Linear system is nearly singular." + ...
                " Recomputing QR decomposition...")
            [Q,R] = qr(A(:,active_set));
            tmp = (b.'*Q).';
            u = R\tmp;
        end    
    else
        u = R\tmp;
        num_linsolve = num_linsolve + 1;
    end
end

x = zeros(n,1); x(active_set) = u;
d = -theta;
ind = find(eq_set);
eq_set(ind(~active_set)) = 0;
end