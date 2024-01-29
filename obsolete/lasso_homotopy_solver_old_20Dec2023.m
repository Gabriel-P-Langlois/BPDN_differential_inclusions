%% Description -- lasso_homotopy_solver
%
%

function [sol_exact_x,sol_exact_p,exact_path] = lasso_homotopy_solver(A,b,tol)
%% Initialization
m = size(A,1); 
n = size(A,2);
tol_minus = 1-tol;

sol_exact_x = zeros(n,m+1);
sol_exact_p = zeros(m,m+1);
exact_path = zeros(m+1,1);

A = [A,-A];

moving_term_1 = (-b.'*A).';
t0 = norm(moving_term_1,inf);
moving_term_1 = moving_term_1/t0;

exact_path(1) = t0;
sol_exact_p(:,1) = -b/t0;
active_set = (moving_term_1 >= tol_minus);



%% The algorithm
shall_continue = true;
i = 0; max_iter = 2*m;

% QR Decomposition, is used
[Q,R] = qr(A);

while(shall_continue && i < max_iter)
    i=i+1;

    %%% 1. Cone projection
    % We compute min_{x} normsq{A_{active}*x + b}

    %K = A(:,active_set);
    %x_active = -K\b;
    Ract = R(:,active_set);
    x_active = -Ract\(Q.'*b);
    

    %%% 2. Compute the direction
    direction = -(Q*(Ract*x_active) + b);


    %%% 3. Compute kick time
    moving_term_2 = (direction.'*A).';%A.'*direction;
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
    sol_exact_p(:,i+1) = sol_exact_p(:,i) + ...
        timestep*direction;
    exact_path(i + 1) = exact_path(i)...
        /(1+exact_path(i)*timestep);

    tmp = zeros(2*n,1); tmp(active_set) = x_active;
    d_active = -tmp(1:end/2) + tmp(end/2+1:end);
    alpha = exact_path(i+1)/exact_path(i);
    sol_exact_x(:,i+1) = alpha*sol_exact_x(:,i) + (1-alpha)*d_active; 

    %%% 6. Compute the active set for the next iteration
    moving_term_1 = moving_term_1 + timestep*moving_term_2;
    active_set = (moving_term_1 >= tol_minus);
end
end

