%% Description -- lasso_homotopy_solver_opt
% Exact homotopy algorithm -- Written by Gabriel Provencher Langlois

% When I remove an index, most of the time the next timestep is too small.
% What is going on here?


function [sol_x,sol_p,sol_t] = ...
    homotopy_exp_27_07_2024(A,b,m,n,tol,display_iterations,run_diagnostics)


%% Initialization
rng('default')
tol_minus = 1-tol;

% Compute t0 and the variable moving_term_1 (= -A.'*p(t0))
bminus = -b;
Atop_times_pk = (bminus.'*A).';
t0 = norm(Atop_times_pk,inf);
Atop_times_pk = Atop_times_pk/t0;

% Compute initial equicorrelation set
equi_set = (abs(Atop_times_pk) >= tol_minus); 

% Create solution paths + initialize
buffer_path = 10*m;
sol_p = zeros(m,buffer_path); 
sol_t = zeros(1,buffer_path);
sol_x = zeros(n,buffer_path);

sol_t(1) = t0;
sol_p(:,1) = bminus/t0;


%% Algorithm
shall_continue = true;
k = 1;
while(shall_continue && k <= buffer_path)
    %%%%% 1. Compute the relevant vectors and matrices
    vector_of_signs = sign(-Atop_times_pk);
    D = diag(vector_of_signs(equi_set));
    K = A(:,equi_set);
    

    %%%%% 2. Solve the LSQ problem
    flag = true;
    while(flag)
        u = (K*D)\b;

        % Retrieve components of the lsq solution that are negative
        ind_u_neg = find(u<-tol); uhat = u(ind_u_neg);

        % Retrieve indices j \in \eps(pk) where uhat_j < 0.
        ind_equi_set = find(equi_set); my_ind = ind_equi_set(ind_u_neg);

        % Retrieve corresponding utilde
        utilde = vector_of_signs(my_ind).*sol_x(my_ind,k);

        % Remove indices violating the other condition
        if(any(abs(utilde) < tol))
            equi_set(my_ind(find(abs(utilde) < tol))) = 0;
            D = diag(vector_of_signs(equi_set));
            K = A(:,equi_set);
        else
            flag = false;
        end
    end

    %%%%% 3. Compute the descent direction
    v = zeros(n,1); v(equi_set) = D*u;
    d = K*(D*u) - b;

    % Compute the variable moving_term_2 (= A.'*d)
    Atop_times_dk = (d.'*A).'; 

    % Compute the set of descent directions (abs(<d,Aej>) > 0)
    active_set_plus = abs(Atop_times_dk) > tol;
    

    %%%%% 4. Compute the kick time
    %%% This is the `smallest' amount of time we can slide down
    % over the possible set of directions before changing the active set.

    % Check if the least square problem returned a solution
    % with negative coefficient.
    if(any(u<-tol))        
        % Compute smallest t for which the convex combination of
        % quantities are zero. Here, tc = factor*tk
        factor = (-uhat./(utilde - uhat));
        tc = max(factor)*sol_t(k);
        timestep_1 = (1/tc - 1/sol_t(k));
    end


    % Check how far we have to go before adding a new index
    if (any(active_set_plus))
        term1 = diag(vector_of_signs)*Atop_times_dk;
        sign_coeffs = sign(term1);
        term2 = -diag(vector_of_signs)*Atop_times_pk;

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
        % Compute final solutions
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

    % if(any(u<-tol) && any(active_set_plus))
    %     if(timestep == timestep_1)
    %         % disp('Removing an index and any(active_set_plus) is nonempty:')
    %         % disp(k)
    %         %equi_set(my_ind) = 0;
    %         equi_set = (abs(Atop_times_pk) >= tol_minus); 
    %     else
    %         equi_set = (abs(Atop_times_pk) >= tol_minus); 
    %     end
    % elseif(any(u<-tol))
    %     % disp('Removing an index and any(active_set_plus) is empty:')
    %     % disp(k)
    %     %equi_set(my_ind) = 0;
    %     equi_set = (abs(Atop_times_pk) >= tol_minus); 
    % else
    %     equi_set = (abs(Atop_times_pk) >= tol_minus); 
    % end



    %%%%% 6. Optionals
    % If enabled, display current iteration
    % if(display_iterations)
    %     disp(['Iteration ',num2str(k),' is complete.'])
    % end

    % If enabled, run diagnostics
    if(run_diagnostics)
        disp(['Iteration: ', num2str(k),' has timestep = ',num2str(timestep)])
        % disp(' ')
        % Compute the residual norm of the NNLS problem
        % min_{u >= 0} ||K*D*u - (b+tk*pk)||^2
        % [~,quantity_1,~] = lsqnonneg(K*D,b + sol_t(k)*sol_p(:,k));
        % disp(['Diagnostic I: Computing min_{u >= 0} ||K*D*u - (b+tk*pk)||^2: ',num2str(quantity_1)]);

        % Display when the solution is strictly positive
        % if(any(u < -tol))
        %     disp('Diagnostic II: At least one component of the NNLS problem is zero.')
        % else
        %     disp('Diagnostic II: All components of the NNLS are strictly positive.')
        % end
        % disp(' ')
        % disp(' ')
    end

    % Increment
    k=k+1;
end
end