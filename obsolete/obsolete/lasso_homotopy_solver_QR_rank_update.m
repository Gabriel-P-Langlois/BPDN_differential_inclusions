%% Description -- lasso_homotopy_solver_opt
%
% Exact homotopy algorithm -- Written by Gabriel Provencher Langlois
%
% This code assumes one feature is update at a time, and use a QR
% decomposition + row rank updates to solve it fast.

function [sol_exact_x,sol_exact_p,exact_path] = ...
    lasso_homotopy_solver_QR_rank_update(A,b,tol,display_iterations)


%% Initialization
m = size(A,1);  n = size(A,2);
sol_exact_p = zeros(m,m+1); sol_exact_x = zeros(n,m+1);
exact_path = zeros(m+1,1);
tol_minus = 1-tol;

bminus = -b; bminust = bminus.';
tmp1 = (bminust*A).';
t0 = norm(tmp1,inf);
tmp1 = tmp1/t0;

moving_term_1 = [tmp1;-tmp1];
active_set = (moving_term_1 >= tol_minus);
present_indices = find(active_set);

% Initial QR decomposition
K = [A(:,active_set(1:n)),-A(:,active_set(n+1:end))];
[Q,R] = qr(K);

% Store first time and first dual solution + options for linear solvers
exact_path(1) = t0;
sol_exact_p(:,1) = bminus/t0;
opts1.LT = true;
opts2.UT = true;

%% The algorithm
shall_continue = true;
i = 0; max_iters = m;

while(shall_continue && i < max_iters)
    i=i+1;

    %%% 1. Cone projection
    % We compute min_{x} normsq{A_{active}*x + b}
    Qtbminus = (bminust*Q).';     % \in \Rm
    d = (Qtbminus.'*R).';

    z = linsolve(R.',d,opts1);
    x_active = linsolve(R,z,opts2);

    % Express solution direction back to Rn coordinates
    tmp = zeros(2*n,1); tmp(active_set) = x_active;
    d_active = -tmp(1:end/2) + tmp(end/2+1:end);

    %%% 2. Compute the descent direction = -(Aeff*xeff+b)
    tmp2 = R*(-x_active);
    direction = (Q*tmp2 + bminus);

    %%% 3. Compute kick time (A.'*direction)
    tmp3 = (direction.'*A).';
    moving_term_2 = [tmp3;-tmp3]; 
    active_set_plus = (moving_term_2 > tol);
    if (any(active_set_plus))
        vec = -(moving_term_1-1)./moving_term_2;
        timestep = min(vec(active_set_plus));
    else % Stop; termination condition satisfied.
        sol_exact_x(:,i+1) = d_active; 
        break;
    end

    %%% 4. Update dual variable and regularization path
    % Update dual solution + time parameter
    sol_exact_p(:,i+1) = sol_exact_p(:,i) + timestep*direction;
    exact_path(i + 1) = exact_path(i)/(1+exact_path(i)*timestep);

    % Update the primal solution using d_active
    alpha = exact_path(i+1)/exact_path(i);
    sol_exact_x(:,i+1) = alpha*sol_exact_x(:,i) + (1-alpha)*d_active; 

    %%% 5. Compute the active set for the next iteration
    moving_term_1 = moving_term_1 + timestep*moving_term_2;
    active_set = (moving_term_1 >= tol_minus);
    next_indices = find(active_set);

    %%% 6. Update QR matrices -- Add a column at the appropriate position
    % Use Givens matrices and code from taken from MATLAB own QR updates;
    % The internal insertCol cannot be accessed
    K = [A(:,active_set(1:n)),-A(:,active_set(n+1:end))];
    [~,pos] = setdiff(next_indices,present_indices);
    R(:,pos+1:i+1) = R(:,pos:i);
    R(:,pos) = Q.'*K(:,pos);
    [Q,R] = matlab.internal.math.insertCol(Q,R,pos);

    present_indices = next_indices;

    %%% 7. If enabled, display current iteration
    if(display_iterations)
        disp(['Iteration ',num2str(i),' is complete.'])
    end
end
end