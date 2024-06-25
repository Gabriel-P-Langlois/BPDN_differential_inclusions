%% Description -- lasso_homotopy_solver_opt
% Exact homotopy algorithm -- Written by Gabriel Provencher Langlois



function [sol_exact_x,sol_exact_p,exact_path] = ...
    lasso_solver_homotopy(A,b,m,n,tol,display_iterations,use_tests)


%% Initialization
rng('default')
tol_minus = 1-tol;

% Compute t0 and the variable moving_term_1 (= -A.'*p(t0))
bminus = -b;
moving_term_1 = (bminus.'*A).';
t0 = norm(moving_term_1,inf);
moving_term_1 = moving_term_1/t0;   % moving_term_1 = A.'*p0;

% Create solution paths + initialize
buffer_path = 5*m;
sol_exact_p = zeros(m,buffer_path); 
exact_path = zeros(1,buffer_path);
sol_exact_x = zeros(n,buffer_path);

exact_path(1) = t0;
sol_exact_p(:,1) = bminus/t0;


%% Algorithm
shall_continue = true;
i = 0;

while(shall_continue)
    i=i+1;

    %%%%% 1. Compute the equicorrelation set to select the active columns,
    % Note: We also compute the relevant signs and store them
    equicorrelation_set = (abs(moving_term_1) >= tol_minus); 
    K = A(:,equicorrelation_set);
    D = diag(sign(-moving_term_1(equicorrelation_set)));

    %%%%% 2. Solve the NNLS problem + store the solution u = D*v
    % Method 1
    v = lsqnonneg(K*D,b);
    u = zeros(n,1); u(equicorrelation_set) = D*v;
    disp("Solution found by NNLS multiplied by matrix of signs")
    disp(D*v)

    % Least-squares
    if(use_tests)
        ntrue = length(K(1,:))/2;
        Atrue = K(:,1:ntrue);
        vtest = Atrue\b;
        disp("Solution by augmented least squares")
        disp(vtest)

        disp("Test 1")
        disp(K\b)

        disp("Sum")
        disp(K\b - D*v)
    end

    %%%%% 3. Compute the descent direction
    descent_direction = (K*u(equicorrelation_set) + bminus);

    % Compute the variable moving_term_2 (= -A.'*d)
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
    exact_path(i+1) = exact_path(i)/(1+exact_path(i)*timestep);

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