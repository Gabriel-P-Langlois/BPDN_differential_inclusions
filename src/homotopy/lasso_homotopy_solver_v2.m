%% Description -- lasso_homotopy_solver_opt
% Exact homotopy algorithm -- Written by Gabriel Provencher Langlois
%
% Essentially an ``auto-updating" least-squares problem

function [sol_exact_x,sol_exact_p,exact_path] = ...
    lasso_homotopy_solver_v2(A,b,tol,display_iterations)


%% Initialization
% Initialize dimensions and (1-tolerance) variable
m = size(A,1);
n = size(A,2);
tol_minus = 1-tol;

% Compute t0 and the variable moving_term_1 (= -A.'*p(t0))
bminus = -b;
moving_term_1 = (bminus.'*A).';
t0 = norm(moving_term_1,inf);
moving_term_1 = moving_term_1/t0;

% Create solution paths + initialize
sol_exact_p = zeros(m,m+1); 
exact_path = zeros(m+1,1);
sol_exact_x = zeros(n,m+1);

exact_path(1) = t0;
sol_exact_p(:,1) = bminus/t0;


%% Algorithm
shall_continue = true;
i = 0;

while(shall_continue && i < m)
    i=i+1;

    %%%%% 1. Compute the equicorrelation set to select the active columns
    equicorrelation_set = (abs(moving_term_1) >= tol_minus); 
    K = A(:,equicorrelation_set);


    %%%%% 2. Solve the corresponding least-squares problem
    % Note: Computational bottleneck (1/2)
    v = K\b;

    % Store the new solution
    u = zeros(n,1);
    u(equicorrelation_set) = v;


    %%%%% 3. Compute the descent direction (= (Aeff*xeff-b))
    descent_direction = (K*u(equicorrelation_set) + bminus);

    % Compute the variable moving_term_2 (= -A.'*d)
    % Note: Computational bottleneck (2/2)
    moving_term_2 = (descent_direction.'*A).'; 

    % Compute the set of descent directions (abs(<d,Aej>) > 0)
    active_set_plus = (abs(moving_term_2) > tol);
    

    %%%%% 4. Compute the kick time
    % Note 1: This is the `smallest' amount of time we can slide down
    % over the possible set of directions before changing the active set.

    % Note 2: If there are no possible set of directions, then 
    % timestep = inf and we have converged.
    if (any(active_set_plus))
        sign_coeffs = sign(moving_term_2(active_set_plus));
        vec = (sign_coeffs-moving_term_1(active_set_plus))./...
            moving_term_2(active_set_plus);
        timestep = min(vec);
    else
        sol_exact_x(:,i+1) = u; 
        break;
    end


    %%%%% 5. Update all solutions and variables
    % Update the dual solution + time parameter
    sol_exact_p(:,i+1) = sol_exact_p(:,i) + timestep*descent_direction;
    exact_path(i + 1) = exact_path(i)/(1+exact_path(i)*timestep);

    % Update the variable moving_term_1 (= -A.'*p(exact_path(i + 1)))
    moving_term_1 = moving_term_1 + timestep*moving_term_2;

    % Update the primal solution
    alpha = exact_path(i+1)/exact_path(i);
    sol_exact_x(:,i+1) = alpha*sol_exact_x(:,i) + (1-alpha)*u; 


    %%%%% OPTIONAL. If enabled, display current iteration
    if(display_iterations)
        disp(['Iteration ',num2str(i),' is complete.'])
    end
end
end