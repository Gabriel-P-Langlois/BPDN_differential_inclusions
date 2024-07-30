function [sol_x,sol_p,sol_t] = ...
    BPDN_exact_algorithm(A,b,tol,disp_output_exact,run_opt_cond_checks)
%% Description -- lasso_homotopy_solver_opt
% Exact BPDN solution using gradient inclusions -- Written by Gabriel Provencher Langlois


%% Initialization
tol_minus = 1-tol;

% Initialize the auxiliary quantities
[m,n] = size(A);
bminus = -b;
Atop_times_bminus = (bminus.'*A).';
t0 = norm(Atop_times_bminus,inf);

% Initialize ``maximum" number of iterations permitted before stopping
% NOTE: Generally will stop around 1.5 to 2.5*m
max_iter = 20*m;

% Initialize the main quantities
sol_x = zeros(n,max_iter);
sol_p = zeros(m,max_iter); sol_p(:,1) = bminus/t0;
sol_t = zeros(1,max_iter); sol_t(1) = t0;

% Compute initial equicorrelation set
Atop_times_pk = Atop_times_bminus/t0;   % Used to speed up the computations
equi_set = (abs(Atop_times_pk) >= tol_minus); 


%% Algorithm
% Run the algorithm until either convergence is achieved or the maximum
% number of iterations permitted is reached.
shall_continue = true;
k = 1;
while(shall_continue && k <= max_iter)
    %%%%% 1. Compute the relevant vectors and matrices
    K = A(:,equi_set);
    vector_of_signs = sign(-Atop_times_pk);
    D = vector_of_signs(equi_set);

    %%%%% 2. Solve the LSQ problem
    while(true)
        u = (K.*vector_of_signs(equi_set).')\b;

        % Retrieve components of the LSQ solution that are negative, and
        % retrieve the corresponding indices in the equicorrelation set.
        ind_u_neg = find(u<-tol); 
        uhat = u(ind_u_neg);

        ind_equi_set = find(equi_set); 
        my_ind = ind_equi_set(ind_u_neg);

        % Retrieve corresponding utilde
        utilde = vector_of_signs(my_ind).*sol_x(my_ind,k);

        % Remove indices violating the other condition
        if(any(abs(utilde) < tol))
            equi_set(my_ind(find(abs(utilde) < tol))) = 0;
            D = vector_of_signs(equi_set);
            K = A(:,equi_set);
        else
            break
        end
    end

    %%%%% 3. Compute the descent direction
    v = zeros(n,1); v(equi_set) = D.*u;
    d = K*(D.*u) + bminus;

    % Compute A.'*d and the set of indices abs(A.'*d) > 0
    Atop_times_dk = (d.'*A).'; 
    active_set_plus = abs(Atop_times_dk) > tol;
    

    %%%%% 4. Compute the kick time
    %%% 4.1. Check if the least square problem returned a solution
    % with negative coefficient.
    if(any(u<-tol))        
        % Compute smallest t for which the convex combination of
        % quantities are zero. Here, tc = factor*tk
        factor = (-uhat./(utilde - uhat));
        tc = max(factor)*sol_t(k);
        timestep_1 = (1/tc - 1/sol_t(k));
    end


    %%% 4.2 Compute thee `smallest' amount of time we can descent over
    % the set of admissible faces. If the set of indices abs(A.'*d) > 0
    % is empty, then timestep = +\infty
    if (any(active_set_plus))
        term1 = vector_of_signs.*Atop_times_dk;
        sign_coeffs = sign(term1);
        term2 = -vector_of_signs.*Atop_times_pk;

        vec = (1+sign_coeffs(active_set_plus).*term2(active_set_plus))./...
            abs(term1(active_set_plus));

        timestep_2 = min(vec);
    end

    % Decide on the timestep to use and update the equicorrelation set
    if(any(u<-tol) && any(active_set_plus))
        timestep = min(timestep_1,timestep_2);
    elseif(any(u<-tol))
        timestep = timestep_1;
    elseif(any(active_set_plus))
        timestep = timestep_2;
    else
        % Convergence achieved --  compute final solutions
        sol_x(:,k+1) = v; 
        sol_p(:,k+1) = sol_p(:,k);

        % Truncate the path and exit
        sol_x(:,k+2:end) = [];
        sol_p(:,k+2:end) = [];
        sol_t(:,k+2:end) = [];
        break;
    end


    %%%%% 5. Update all solutions and variables
    % Update the dual solution + time parameter
    sol_p(:,k+1) = sol_p(:,k) + timestep*d;
    sol_t(k+1) = sol_t(k)/(1+sol_t(k)*timestep);

    % Update the primal solution
    alpha = sol_t(k+1)/sol_t(k);
    sol_x(:,k+1) = alpha*sol_x(:,k) + (1-alpha)*v; 

    % Compute next equicorrelation set
    Atop_times_pk = Atop_times_pk + timestep*Atop_times_dk;
    equi_set = (abs(Atop_times_pk) >= tol_minus); 


    %%%%% 6. (Optional) Display information about the current iteration
    % If enabled, display the current iteration and the timestep
    % used in the calculation.
    if(disp_output_exact)
        disp(['Iteration: ', num2str(k),' has t = ',num2str(sol_t(k)) ' and timestep = ',num2str(timestep)])
    end

    % If enabled, display some diagnostics
    % WARNING: Enabling this will considerable slow down the algorithm
    if(run_opt_cond_checks)
        disp(' ')
        % Compute the residual norm of the NNLS problem
        % min_{u >= 0} ||K*D*u - (b+tk*pk)||^2
        [~,quantity_1,~] = lsqnonneg(K*D,b + sol_t(k)*sol_p(:,k));
        disp(['Diagnostic I: Computing min_{u >= 0} ||K*D*u - (b+tk*pk)||^2: ',num2str(quantity_1)]);

        % Display when the solution is strictly positive
        if(any(u < -tol))
            disp('Diagnostic II: At least one component of the NNLS problem is zero.')
        else
            disp('Diagnostic II: All components of the NNLS are strictly positive.')
        end
        disp(' ')
        disp(' ')
    end

    %%%%% 7. Increment
    k=k+1;
end

% Display warning if max_iter has been reached
% Note: May possibly be reached for some contorted matrix A.
if(k == max_iter)
    disp('Warning: k = 100*m iterations reached!')
end
end