function [x,p,nb_iterations] = lasso_dual_diff_incl(p,A,b,t,tol)

% Compute size of matrix A(:,selection_set)
% Note: factor = ratio*A_vecnorm(selection_set);
n = size(A,2);

% Initial values
lambda = p;                             % Initial value
mv_prod1 = lambda.'*A;                  % Compute <aj,pk>

% Parameters for the algorithm
shall_continue = true;
tol_minus = 1-tol;
nb_iterations = 0;

% The algorithm
while (shall_continue)
    %%% 1. Compute active set
    v_active = [mv_prod1,-mv_prod1];
    active_set = (v_active.' >= tol_minus);

    %%% 2. Compute direction
    K = [A(:,active_set(1:n)),-A(:,active_set(n+1:end))];
    [ME, NU] = size(K);

    direction = -b - t*lambda;
    x_nnls = nnls_for_paths_fast(K,direction,ME,NU);
    direction = direction - K*x_nnls;
    
    %%% 3. Compute kick time
    % Compute S_Cˆ+(lambda) set
    mv_prod2 = (direction.'*A).';%A.'*direction;
    v_direction = [mv_prod2;-mv_prod2];
    active_set_plus = (v_direction > tol);
    asp_check = any(active_set_plus);

    % Verify if active_set_plus is empty. Stop if it is, else continue.
    if (asp_check)
        vec = (1-v_active)./v_direction.';
        timestep = min(vec(active_set_plus));
        % Safety check: t must be non negative
        if (timestep < 0) 
            timestep = 0;
        end
    else
        timestep = realmax;
    end

    %%% 4. Check termination condition
    if(timestep==realmax || timestep == 0) % Stop; we are done.
        nb_iterations=nb_iterations+1;
        break;
    else
        if(t~=0)
            c = (1/t)*(1-exp(-t*timestep));
            lambda_plus = lambda + c*direction; 
        else
            c = 1;
            lambda_plus = lambda + direction; 
        end
        if (norm(lambda_plus-lambda) < tol) % Convergence achieved!
            lambda = lambda_plus;
            nb_iterations=nb_iterations+1;
            break;
        end
        
        %%% 5. Update variables and proceed to the next iteration
        % Compute the scalar product <aj,lambda_plus>
        mv_prod1 = mv_prod1 + c*mv_prod2.';
        
        % Update dual variable
        lambda = lambda_plus;
        nb_iterations=nb_iterations+1;        
    end
end

%%% 5. Return primal and dual solution
tmp = zeros(2*n,1); tmp(active_set) = x_nnls;
x = -tmp(1:end/2) + tmp(end/2+1:end);
p = lambda;
end