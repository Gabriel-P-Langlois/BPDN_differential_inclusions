%% Description -- lasso_homotopy_solver_opt
%
% Exact homotopy algorithm
%
% This is an optimized version of the code from Ciril and others
% that is adapted to solve the Lasso problem in its entirety.
%
% 
% Written by Gabriel Provencher Langlois
%
% Notes:
%
% This piece of code assumes the length of the path is exactly equal to
% m+1... It may fail to work if the path happens to be longer (we remove
% elements from the active set)... but this will likely happen only very
% highly ill-posed matrices and matrices with integer entries.

function [sol_exact_x,sol_exact_p,exact_path] = lasso_homotopy_solver_QR(A,b,tol)


%% Initialization
m = size(A,1); 
n = size(A,2);
sol_exact_x = zeros(n,m+1);
sol_exact_p = zeros(m,m+1);
exact_path = zeros(m+1,1);
tol_minus = 1-tol;

bminus = -b;
bminust = bminus.';
tmp1 = (bminust*A).';
t0 = norm(tmp1,inf);
tmp1 = tmp1/t0;

moving_term_1 = [tmp1;-tmp1];
active_set = (moving_term_1 >= tol_minus);

% QR Decomposition
[Q,R] = qr(A);
Qtbminus = (bminust*Q).'; %-Q.'*b;
RtQtbminus = ((bminust*Q)*R).';   % -R.'*Q.'*b;

% Store first time and first dual solution
exact_path(1) = t0;
sol_exact_p(:,1) = bminus/t0;


%% The algorithm
shall_continue = true;
i = 0; max_iters = m;

while(shall_continue && i < max_iters)
    i=i+1;

    %%% 1. Cone projection
    % We compute min_{x} normsq{A_{active}*x + b}
    Ract = [R(:,active_set(1:n)),-R(:,active_set(n+1:end))];
    x_active = Ract\Qtbminus;

    %%% 2. Compute the direction
    tmp2 = Ract*(-x_active);
    direction = (Q*tmp2 + bminus);

    %%% 3. Compute kick time (A.'*direction)
    tmp3 = (tmp2.'*R).';
    tmp4 = RtQtbminus + tmp3;
    moving_term_2 = [tmp4;-tmp4]; 
    active_set_plus = (moving_term_2 > tol);
    if (any(active_set_plus))
        vec = -(moving_term_1-1)./moving_term_2;
        timestep = min(vec(active_set_plus));

        % Safety check: t must be non-negative.
        if (timestep < 0) 
            tmp = zeros(2*n,1); tmp(active_set) = x_active;
            d_active = -tmp(1:end/2) + tmp(end/2+1:end);
            sol_exact_x(:,i+1) = d_active; 
            break;
        end

    else % Stop; termination condition satisfied.
        tmp = zeros(2*n,1); tmp(active_set) = x_active;
        d_active = -tmp(1:end/2) + tmp(end/2+1:end);
        sol_exact_x(:,i+1) = d_active; 
        break;
    end


    %%% 5. Update dual variable and regularization path
    % Update dual solution + time parameter
    sol_exact_p(:,i+1) = sol_exact_p(:,i) + timestep*direction;
    exact_path(i + 1) = exact_path(i)/(1+exact_path(i)*timestep);

    % Update primal solution
    tmp = zeros(2*n,1); tmp(active_set) = x_active;
    d_active = -tmp(1:end/2) + tmp(end/2+1:end);
    alpha = exact_path(i+1)/exact_path(i);
    sol_exact_x(:,i+1) = alpha*sol_exact_x(:,i) + (1-alpha)*d_active; 

    %%% 6. Compute the active set for the next iteration
    moving_term_1 = moving_term_1 + timestep*moving_term_2;
    active_set = (moving_term_1 >= tol_minus);
end
end

